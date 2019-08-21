use std::cmp::min;
use std::io::Write;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::mpsc;
use std::thread::spawn;
use bincode::serialize;
use bincode::deserialize;
use commons::Model;
use commons::ModelSig;
use commons::channel::Sender;
use commons::io::create_bufwriter;
use commons::io::load_s3 as io_load_s3;
use commons::io::write_s3 as io_write_s3;
use commons::performance_monitor::PerformanceMonitor;
use tree::Tree;
use tmsn::network::start_network_only_recv;


pub const REGION:   &str = "us-east-1";
pub const BUCKET:   &str = "tmsn-cache2";
pub const S3_PATH_MODELS:  &str = "sparrow-models/";
pub const MODEL_FILENAME: &str = "model.bin";
pub const S3_PATH_ASSIGNS:  &str = "sparrow-assigns/";
pub const ASSIGN_FILENAME: &str = "assign.bin";

pub fn start_model_sync(
    init_tree: Tree,
    name: String,
    remote_ips: Vec<String>,
    port: u16,
    next_model: Sender<Model>,
    default_gamma: f32,
    current_sample_version: Arc<RwLock<usize>>,
    exp_name: String,
) {
    upload_model(&init_tree, &"init".to_string(), default_gamma, &exp_name);
    let (local_s, local_r): (mpsc::Sender<ModelSig>, mpsc::Receiver<ModelSig>) =
        mpsc::channel();
    start_network_only_recv(name.as_ref(), &remote_ips, port, local_s);
    spawn(move || {
        model_sync_main(
            init_tree, remote_ips.len(), local_r, next_model, default_gamma,
            current_sample_version, &exp_name);
    });
}


// Worker download models
pub fn download_model(exp_name: &String) -> Option<(Model, String, f32)> {
    // debug!("sampler, start, download model");
    let s3_path = format!("{}/{}", exp_name, S3_PATH_MODELS);
    let ret = io_load_s3(REGION, BUCKET, s3_path.as_str(), MODEL_FILENAME);
    // debug!("sampler, finished, download model");
    if ret.is_none() {
        debug!("sample, download model, failed");
        return None;
    }
    let (data, code) = ret.unwrap();
    if code == 200 {
        // debug!("sample, download model, succeed");
        Some(deserialize(&data).unwrap())
    } else {
        debug!("sample, download model, failed with return code {}", code);
        None
    }
}


pub fn download_assignments(exp_name: &String) -> Option<Vec<Option<usize>>> {
    let s3_path = format!("{}/{}", exp_name, S3_PATH_ASSIGNS);
    let ret = io_load_s3(REGION, BUCKET, s3_path.as_str(), ASSIGN_FILENAME);
    // debug!("model sync, finished, download assignments");
    if ret.is_none() {
        // debug!("model sync, download assignments, failed");
        return None;
    }
    let (data, code) = ret.unwrap();
    if code == 200 {
        // debug!("model sync, download assignments, succeed");
        Some(deserialize(&data).unwrap())
    } else {
        debug!("model sync, download assignments, failed with return code {}", code);
        None
    }
}


// Server upload models
fn upload_model(model: &Model, sig: &String, gamma: f32, exp_name: &String) -> bool {
    let data = (model.clone(), sig.clone(), gamma);
    let s3_path = format!("{}/{}", exp_name, S3_PATH_MODELS);
    io_write_s3(REGION, BUCKET, s3_path.as_str(), MODEL_FILENAME, &serialize(&data).unwrap())
}


fn model_sync_main(
    model: Model,
    num_machines: usize,
    receiver: mpsc::Receiver<ModelSig>,
    next_model_sender: Sender<Model>,
    default_gamma: f32,
    current_sample_version: Arc<RwLock<usize>>,
    exp_name: &String,
) {
    // TODO: make duration and fraction config parameters
    const DURATION: f32 = 15.0;
    const FRACTION: f32 = 0.1;
    // State variables
    // TODO: Infer all state variables based on `model`
    let mut model = model;
    let mut model_sig = "init".to_string();
    let mut gamma = default_gamma.clone();
    let mut shrink_factor = 0.9;
    let mut last_condition = 0;  // -1 => TOO_SLOW, 1 => TOO_FAST
    let mut node_status = vec![(0, default_gamma, None); model.tree_size];
    let mut worker_assign = vec![None; num_machines];
    // Performance variables
    let mut global_timer = PerformanceMonitor::new();
    let mut timer = PerformanceMonitor::new();
    let mut total_packets = 0;
    let mut rejected_packets = 0;
    let mut failed_searches = 0;
    let mut num_updates_packs = vec![0; num_machines];
    let mut num_updates_rejs  = vec![0; num_machines];
    let mut num_updates_nodes = vec![0; num_machines];
    let mut node_sum_gamma_sq = vec![0.0];
    let mut node_timestamp = vec![0.0];
    timer.start();
    global_timer.start();
    // initialize state variables based on `model`
    for i in 0..min(node_status.len(), worker_assign.len()) {
        node_status[i] = (model.depth[i], gamma, Some(i));
        worker_assign[i] = Some(i);
    }
    loop {
        // adjust gamma
        if timer.get_duration() >= DURATION {
            let mut current_condition = 0;
            let threshold = min(num_machines, node_status.len()) as f32 * 0.5;
            if failed_searches >= threshold as usize {
                // alternative: if total_packets == 0
                current_condition = -1;
            } else if (rejected_packets as f32) / (total_packets as f32) >= FRACTION {
                current_condition = 1;
            }
            if current_condition == -1 {
                if last_condition == 1 {
                    shrink_factor = (1.0 + shrink_factor) / 2.0;
                } else if last_condition == -1 {
                    shrink_factor = (0.8 + shrink_factor) / 2.0;
                }
            }
            let old_gamma = gamma;
            match current_condition {
                1  => gamma = gamma / shrink_factor,
                -1 => gamma = gamma * shrink_factor,
                _  => {},
            }
            if current_condition != 0 {
                if upload_model(&model, &model_sig, gamma, exp_name) {
                    debug!("model_manager, broadcast gamma, {}, {}, {}, {}",
                           current_condition, gamma, shrink_factor, old_gamma);
                } else {
                    debug!("model_manager, failed gamma broadcast, {}, {}, {}, {}",
                           current_condition, gamma, shrink_factor, old_gamma);
                }
                failed_searches = 0;
            }
            let packs_stats: Vec<String> = num_updates_packs.iter().map(|t| t.to_string()).collect();
            let rejs_stats: Vec<String>  = num_updates_rejs.iter().map(|t| t.to_string()).collect();
            let nodes_stats: Vec<String> = num_updates_nodes.iter().map(|t| t.to_string()).collect();
            debug!("model_manager, status update, {}, {}, {}, {}, {}, {}, {}, {}",
                   gamma, shrink_factor, failed_searches, total_packets, rejected_packets,
                   packs_stats.join(", "), rejs_stats.join(", "), nodes_stats.join(", "));
            last_condition = current_condition;
            total_packets = 0;
            rejected_packets = 0;
            num_updates_packs.iter_mut().for_each(|t| *t = 0);
            num_updates_rejs.iter_mut().for_each(|t| *t = 0);
            num_updates_nodes.iter_mut().for_each(|t| *t = 0);
            timer.reset();
            timer.start();
        }
        update_assignments(
            &mut node_status, &mut worker_assign, gamma, exp_name,
            &mut node_sum_gamma_sq, &mut node_timestamp, global_timer.get_duration());
        let packet = receiver.try_recv();
        if packet.is_err() {
            continue;
        }
        let (patch, remote_gamma, sample_version, old_sig, new_sig) = packet.unwrap();
        let machine_name = {
            let t: Vec<&str> = new_sig.rsplitn(2, '_').collect();
            t[1].to_string()
        };
        let machine_id: usize = {
            let t: Vec<&str> = machine_name.rsplitn(2, '_').collect();
            t[0].parse::<usize>().unwrap() % num_machines
        };
        num_updates_packs[machine_id] += 1;
        total_packets += 1;
        if old_sig != model_sig {
            debug!("model_manager, reject for base model mismatch, {}, {}", model_sig, old_sig);
            num_updates_rejs[machine_id] += 1;
            rejected_packets += 1;
            continue;
        }
        let current_version: usize = {
            *(current_sample_version.read().unwrap())
        };
        if sample_version != current_version {
            debug!("model_manager, reject for sample version mismatch, {}, {}",
                   sample_version, current_version);
            num_updates_rejs[machine_id] += 1;
            rejected_packets += 1;
            continue;
        }
        if patch.size == 0 {
            if worker_assign[machine_id].is_some() {
                let node_id = worker_assign[machine_id].unwrap();
                node_status[node_id] = (node_status[node_id].0, remote_gamma, None);
                worker_assign[machine_id] = None;
                failed_searches += 1;
                let duration = global_timer.get_duration() - node_timestamp[node_id];
                debug!("model_manager, empty, {}, {}, {}, {}, {}, {}, {}",
                        machine_id, node_id, remote_gamma, failed_searches,
                        node_sum_gamma_sq[node_id], duration,
                        node_sum_gamma_sq[node_id] / duration);
            }
        } else {
            let new_nodes_depth = model.append_patch(&patch, remote_gamma, old_sig == "init");
            num_updates_nodes[machine_id] += patch.size;
            model_sig = new_sig;
            next_model_sender.send(model.clone());
            let node_id = {
                if worker_assign[machine_id].is_some() {
                    worker_assign[machine_id].unwrap()
                } else {
                    0
                }
            };
            let (count_new, count_updates) = patch.is_new.iter().fold(
                (0, 0), |(new, old), t| { if *t { (new + 1, old) } else { (new, old + 1) } });
            debug!("model_manager, new updates, {}, {}, {}, {}, {}, {}",
                    machine_id, node_id, remote_gamma, patch.size, count_new, count_updates);
            if upload_model(&model, &model_sig, gamma, exp_name) {
                debug!("model_manager, accept, {}, {}, {}, {}",
                       old_sig, model_sig, machine_name, patch.size);
            } else {
                debug!("model_manager, upload failed, {}, {}, {}, {}",
                       old_sig, model_sig, machine_name, patch.size);
            }
            handle_persistent(&model, model.size(), global_timer.get_duration());
            for depth in new_nodes_depth {
                node_status.push((depth, default_gamma, None));
                node_sum_gamma_sq.push(0.0);
                node_timestamp.push(0.0);
            }
            node_sum_gamma_sq[node_id] += remote_gamma * remote_gamma * patch.size as f32;
        }
    }
}


fn update_assignments(
    node_status: &mut Vec<(usize, f32, Option<usize>)>,
    worker_assign: &mut Vec<Option<usize>>,
    gamma: f32,
    exp_name: &String,
    node_sum_gamma_sq: &mut Vec<f32>,
    node_timestamp: &mut Vec<f32>,
    cur_timestamp: f32,
) {
    let mut did_something = true;
    let mut num_updates = 0;
    while did_something {
        did_something = false;
        let mut valid_worker = 0;
        while valid_worker < worker_assign.len() && worker_assign[valid_worker].is_some() {
            valid_worker += 1;
        }
        if valid_worker < worker_assign.len() {
            let mut min_depth = 9999;
            let mut node = 0;
            for i in 0..node_status.len() {
                let (depth, old_gamma, status) = node_status[i];
                if status.is_none() && old_gamma > gamma && depth < min_depth {
                    node = i;
                    min_depth = depth;
                }
            }
            if min_depth < 9999 {
                node_status[node] = (min_depth, gamma, Some(valid_worker));
                node_sum_gamma_sq[node] = 0.0;
                node_timestamp[node] = cur_timestamp;
                worker_assign[valid_worker] = Some(node);
                did_something = true;
                num_updates += 1;
                info!("model-manager, assign, {}, {}", valid_worker, node);
            }
        }
    }
    if num_updates > 0 {
        debug!("assign updates, {}", num_updates);
        let s3_path = format!("{}/{}", exp_name, S3_PATH_ASSIGNS);
        io_write_s3(REGION, BUCKET, s3_path.as_str(), ASSIGN_FILENAME,
                    &serialize(worker_assign).unwrap());
    }
}


fn handle_persistent(model: &Model, iteration: usize, timestamp: f32) {
    let json = serde_json::to_string(&(timestamp, iteration, model)).expect(
        "Local model cannot be serialized."
    );
    let mut file_buffer = create_bufwriter(
        &format!("models/model_{}-v{}.json", iteration, iteration));
    file_buffer.write(json.as_ref()).unwrap();
}
