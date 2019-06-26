use std::io::Write;
use std::sync::mpsc;
use std::thread::spawn;
use bincode::serialize;
use bincode::deserialize;
use commons::Model;
use commons::ModelSig;
use commons::channel::bounded;
use commons::channel::Sender;
use commons::channel::Receiver;
use commons::io::create_bufwriter;
use commons::io::load_s3 as io_load_s3;
use commons::io::write_s3 as io_write_s3;
use commons::performance_monitor::PerformanceMonitor;
use tree::Tree;
use tmsn::network::start_network_only_recv;


pub const FILENAME: &str = "model.bin";
pub const REGION:   &str = "us-east-1";
pub const BUCKET:   &str = "tmsn-cache2";
pub const S3_PATH:  &str = "sparrow-models/";



pub fn start_model_sync(
    tree_size: usize,
    name: String,
    remote_ips: &Vec<String>,
    port: u16,
    next_model: Sender<Model>,
    default_gamma: f32,
) {
    // TODO: make K a config parameter
    const K: usize = 5;
    let (local_s, local_r): (mpsc::Sender<ModelSig>, mpsc::Receiver<ModelSig>) =
        mpsc::channel();
    let (bounded_local_s, bounded_local_r) = bounded(K, "model-patches");
    start_network_only_recv(name.as_ref(), remote_ips, port, local_s);
    upload_model(&Tree::new(tree_size, 0.0, 0.0), &"".to_string(), default_gamma);
    debug!("Starting the receive models module");
    let bounded_sender = bounded_local_s.clone();
    spawn(move || {
        loop {
            bounded_sender.send(local_r.recv().unwrap());
        }
    });
    let num_machines = remote_ips.len();
    spawn(move || {
        receive_models(num_machines, bounded_local_r, bounded_local_s, next_model, default_gamma);
    });
}


// Worker download models
pub fn download_model() -> Option<(Model, String, f32)> {
    debug!("sampler, start, download model");
    let ret = io_load_s3(REGION, BUCKET, S3_PATH, FILENAME);
    debug!("sampler, finished, download model");
    if ret.is_none() {
        debug!("sample, download model, failed");
        return None;
    }
    let (data, code) = ret.unwrap();
    if code == 200 {
        debug!("sample, download model, succeed");
        Some(deserialize(&data).unwrap())
    } else {
        debug!("sample, download model, failed with return code {}", code);
        None
    }
}


// Server upload models
fn upload_model(model: &Model, sig: &String, gamma: f32) -> bool {
    let data = (model.clone(), sig.clone(), gamma);
    io_write_s3(REGION, BUCKET, S3_PATH, FILENAME, &serialize(&data).unwrap())
}


fn receive_models(
    num_machines: usize,
    receiver: Receiver<ModelSig>,
    sender:   Sender<ModelSig>,
    next_model_sender: Sender<Model>,
    default_gamma: f32,
) {
    // TODO: make duration and fraction config parameters
    const DURATION: f32 = 5.0;
    const FRACTION: f32 = 0.1;
    let mut model_sig = "".to_string();
    let mut model = Tree::new(1, 0.0, 0.0);
    // let mut gamma = HashMap::new();
    let mut global_timer = PerformanceMonitor::new();
    let mut timer = PerformanceMonitor::new();
    let mut gamma = default_gamma;
    let mut shrink_factor = 0.9;
    let mut last_condition = 0;  // -1 => TOO_SLOW, 1 => TOO_FAST
    let mut total_packets = 0;
    let mut rejected_packets = 0;
    let mut num_updates_packs = vec![0; num_machines + 1];
    let mut num_updates_rejs  = vec![0; num_machines + 1];
    let mut num_updates_nodes = vec![0; num_machines + 1];
    timer.start();
    global_timer.start();
    loop {
        if timer.get_duration() >= DURATION {
            let mut current_condition = 0;
            if total_packets == 0 {
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
            match current_condition {
                1  => gamma = gamma / shrink_factor,
                -1 => gamma = gamma * shrink_factor,
                _  => {},
            }
            if current_condition != 0 {
                if upload_model(&model, &model_sig, gamma) {
                    debug!("model_manager, broadcast gamma, {}, {}, {}",
                           current_condition, gamma, shrink_factor);
                } else {
                    debug!("model_manager, failed gamma broadcast, {}, {}, {}",
                           current_condition, gamma, shrink_factor);
                }
            }
            let packs_stats: Vec<String> = num_updates_packs.iter().map(|t| t.to_string()).collect();
            let rejs_stats: Vec<String>  = num_updates_rejs.iter().map(|t| t.to_string()).collect();
            let nodes_stats: Vec<String> = num_updates_nodes.iter().map(|t| t.to_string()).collect();
            debug!("model_manager, status update, {}, {}, {}, {}, {}, {}, {}",
                   gamma, shrink_factor, total_packets, rejected_packets,
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
        let packet = receiver.try_recv();
        if packet.is_none() {
            continue;
        }
        let (patch, remote_gamma, old_sig, new_sig) = packet.unwrap();
        let machine_name = {
            let t: Vec<&str> = new_sig.rsplitn(2, '_').collect();
            t[1].to_string()
        };
        let machine_id: usize = {
            let t: Vec<&str> = machine_name.rsplitn(2, '_').collect();
            t[0].parse().unwrap()
        };
        num_updates_packs[machine_id] += 1;
        total_packets += 1;
        if old_sig != model_sig {
            debug!("model_manager, reject for base model mismatch, {}, {}", model_sig, old_sig);
            num_updates_rejs[machine_id] += 1;
            rejected_packets += 1;
            continue;
        }
        model.append_patch(&patch, remote_gamma, old_sig == "");
        num_updates_nodes[machine_id] += patch.size;
        model_sig = new_sig;
        next_model_sender.send(model.clone());
        if upload_model(&model, &model_sig, gamma) {
            debug!("model_manager, accept, {}, {}, {}, {}",
                   old_sig, model_sig, machine_name, patch.size);
        } else {
            debug!("model_manager, upload failed, {}, {}, {}, {}",
                   old_sig, model_sig, machine_name, patch.size);
        }
        handle_persistent(&model, model.size, global_timer.get_duration());
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