use std::cmp::max;
use std::cmp::min;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::mpsc;
use std::thread::spawn;

use tmsn::network::start_network_only_recv;

use commons::Model;
use commons::ModelSig;
use commons::channel::Sender;
use commons::tree::Tree;
use commons::performance_monitor::PerformanceMonitor;

use commons::network::upload_assignments;
use commons::network::upload_model;
use commons::persistent_io::write_model;


pub fn start_model_sync(
    init_tree: Tree,
    name: String,
    num_iterations: usize,
    num_trees: usize,
    max_depth: usize,
    remote_ips: Vec<String>,
    port: u16,
    next_model: Sender<Model>,
    default_gamma: f32,
    min_gamma: f32,
    current_sample_version: Arc<RwLock<usize>>,
    node_counts: Arc<RwLock<Vec<u32>>>,
    exp_name: String,
    sampler_state: Arc<RwLock<bool>>,
) {
    upload_model(&init_tree, &"init".to_string(), default_gamma, default_gamma, &exp_name);
    let (local_s, local_r): (mpsc::Sender<ModelSig>, mpsc::Receiver<ModelSig>) =
        mpsc::channel();
    start_network_only_recv(name.as_ref(), &remote_ips, port, local_s);
    spawn(move || {
        model_sync_main(
            init_tree, num_iterations, num_trees, max_depth,
            remote_ips.len(), local_r, next_model,
            default_gamma, min_gamma,
            current_sample_version, node_counts, &exp_name, sampler_state);
    });
}


fn model_sync_main(
    model: Model,
    num_iterations: usize,
    max_num_trees: usize,
    max_depth: usize,
    num_machines: usize,
    receiver: mpsc::Receiver<ModelSig>,
    next_model_sender: Sender<Model>,
    default_gamma: f32,
    min_gamma: f32,
    current_sample_version: Arc<RwLock<usize>>,
    node_counts: Arc<RwLock<Vec<u32>>>,
    exp_name: &String,
    sampler_state: Arc<RwLock<bool>>,
) {
    // TODO: make duration and fraction config parameters
    const FRACTION: f32 = 0.1;
    // State variables
    // TODO: Infer all state variables based on `model`
    let mut model = model;
    let mut model_sig = "init".to_string();
    let mut gamma = default_gamma.clone();
    let mut root_gamma = default_gamma.clone();
    let mut shrink_factor = 0.9;
    let mut last_condition = 0;  // -1 => TOO_SLOW, 1 => TOO_FAST
    // Node status: ((depth, num_recent_success), last_failed_gamma, current_scanner_id)
    let mut node_status = vec![((0, 0), 1.0, None); model.tree_size];
    let mut worker_assign = vec![None; num_machines];
    let mut avail_nodes = 0;
    // Performance variables
    let mut global_timer = PerformanceMonitor::new();
    let mut timer = PerformanceMonitor::new();
    let mut total_packets = 0;
    let mut nonempty_packets = 0;
    let mut rejected_packets = 0;
    let mut rejected_packets_model = 0;
    let mut failed_searches = 0;
    let mut num_updates_packs = vec![0; num_machines];
    let mut num_updates_rejs  = vec![0; num_machines];
    let mut num_updates_nodes = vec![0; num_machines];
    let mut node_sum_gamma_sq = vec![0.0; model.tree_size];
    let mut node_timestamp = vec![0.0; model.tree_size];
    timer.start();
    global_timer.start();
    // initialize state variables based on `model`
    for i in 0..min(node_status.len(), worker_assign.len()) {
        let (depth, num_child) = (model.depth[i], model.children[i].len());
        if depth < max_depth && (i == 0 && num_child < max_num_trees || i > 0 && num_child <= 1) {
            avail_nodes += 1;
        }
        node_status[i] = ((depth, num_child), 1.0, Some(i));
        worker_assign[i] = Some(i);
    }
    for i in node_status.len()..worker_assign.len() {
        worker_assign[i] = Some(0);
    }
    let mut last_timestamp = global_timer.get_duration();
    let mut avg_empty_rate = 2.0;
    let mut state = true;
    while state && gamma >= min_gamma && (num_iterations <= 0 || model.size() < num_iterations) {
        state = {
            let t = sampler_state.read().unwrap();
            *t
        };
        // adjust gamma
        let mut verbose = false;
        // wait for enough number of packets, and adjust gamma accordingly
        if total_packets >= max(5, min(avail_nodes, num_machines)) {
            let mut current_condition = 0;
            // TODO: set decrease gamma threshold a parameter
            let empty_rate = 1.0 - (nonempty_packets as f32) / (max(1, total_packets) as f32);
            avg_empty_rate = {
                if avg_empty_rate > 1.0 {
                    empty_rate
                } else {
                    let alpha = 0.1;
                    (1.0 - alpha) * avg_empty_rate + alpha * empty_rate
                }
            };
            if empty_rate >= 1.0 - FRACTION {
                // alternative: if total_packets == 0
                current_condition = -1;
            } else if empty_rate <= FRACTION ||
                    (rejected_packets_model as f32) / (max(1, nonempty_packets) as f32) >= 0.5 {
                current_condition = 1;
            }
            if current_condition != 0 && last_condition != 0 {
                if current_condition != last_condition {
                    shrink_factor = (1.0 + shrink_factor) / 2.0;
                } else {
                    shrink_factor = (0.8 + shrink_factor) / 2.0;
                }
            }
            let old_gamma = gamma;
            match current_condition {
                1  => gamma = gamma / shrink_factor,
                -1 => gamma = gamma * shrink_factor,
                _  => {},
            }
            if current_condition == 1 {
                // allow re-assessing all tree nodes if we increase gamma
                for node_id in 0..node_status.len() {
                    let (status, _remote_gamma, assignment) = node_status[node_id];
                    node_status[node_id] = (status, 1.0, assignment);
                }
            }
            if current_condition != 0 {
                if upload_model(&model, &model_sig, gamma, root_gamma, exp_name) {
                    debug!("model_manager, broadcast gamma, {}, {}, {}, {}, {}",
                           current_condition, gamma, root_gamma, shrink_factor, old_gamma);
                } else {
                    debug!("model_manager, failed gamma broadcast, {}, {}, {}, {}, {}",
                           current_condition, gamma, root_gamma, shrink_factor, old_gamma);
                }
                failed_searches = node_status.iter().map(|(_, last_gamma, avail)| {
                    if avail.is_none() && *last_gamma <= gamma {
                        1
                    } else {
                        0
                    }
                }).sum();
            }
            let packs_stats: Vec<String> = num_updates_packs.iter().map(|t| t.to_string()).collect();
            let rejs_stats: Vec<String>  = num_updates_rejs.iter().map(|t| t.to_string()).collect();
            let nodes_stats: Vec<String> = num_updates_nodes.iter().map(|t| t.to_string()).collect();
            debug!("model_manager, status update, \
                    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {:?}, {:?}",
                   gamma, root_gamma, shrink_factor, failed_searches,
                   total_packets, nonempty_packets, rejected_packets, rejected_packets_model,
                   packs_stats.join(", "), rejs_stats.join(", "), nodes_stats.join(", "),
                   node_status, worker_assign);
            last_condition = current_condition;
            total_packets = 0;
            nonempty_packets = 0;
            rejected_packets = 0;
            rejected_packets_model = 0;
            num_updates_packs.iter_mut().for_each(|t| *t = 0);
            num_updates_rejs.iter_mut().for_each(|t| *t = 0);
            num_updates_nodes.iter_mut().for_each(|t| *t = 0);
            timer.reset();
            timer.start();
            verbose = true;
        }
        update_assignments(
            &mut node_status, &mut worker_assign, gamma, root_gamma, max_depth,
            max_num_trees, exp_name,
            &model.parent, &mut node_sum_gamma_sq, &mut node_timestamp, global_timer.get_duration());
        let packet = receiver.try_recv();
        if packet.is_err() {
            if verbose {
                debug!("model_manager, packet error, {:?}", packet);
            }
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
        let node_id = {
            if worker_assign[machine_id].is_some() {
                worker_assign[machine_id].unwrap()
            } else {
                0
            }
        };
        let node_count = {
            let c = node_counts.read().unwrap();
            if node_id >= c.len() {
                0
            } else {
                c[node_id]
            }
        };
        if patch.size == 0 {
            if worker_assign[machine_id].is_some() {
                let (depth, num_child) = node_status[node_id].0;
                node_status[node_id] = ((depth, num_child), remote_gamma, None);
                worker_assign[machine_id] = None;
                failed_searches += 1;
                let duration = global_timer.get_duration() - node_timestamp[node_id];
                debug!("model_manager, empty, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                        machine_id, node_id, node_count, model.depth[node_id], remote_gamma,
                        failed_searches, node_sum_gamma_sq[node_id], duration,
                        node_sum_gamma_sq[node_id] / duration);
                if node_id == 0 {
                    let old_gamma = root_gamma;
                    root_gamma *= 0.8;
                    if upload_model(&model, &model_sig, gamma, root_gamma, exp_name) {
                        debug!("model_manager, broadcast gamma, {}, {}, {}, {}, {}",
                               -1, gamma, root_gamma, shrink_factor, old_gamma);
                    } else {
                        debug!("model_manager, failed gamma broadcast, {}, {}, {}, {}, {}",
                               -1, gamma, root_gamma, shrink_factor, old_gamma);
                    }
                }
            } else {
                debug!("model_manager, empty with no assignment, {}", machine_id);
            }
            continue;
        }
        nonempty_packets += 1;
        if old_sig != model_sig {
            debug!("model_manager, reject for base model mismatch, {}, {}", model_sig, old_sig);
            num_updates_rejs[machine_id] += 1;
            rejected_packets += 1;
            rejected_packets_model += 1;
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
        // accept the package
        let new_nodes_depth = model.append_patch(&patch, remote_gamma, old_sig == "init");
        num_updates_nodes[machine_id] += patch.size;
        model_sig = new_sig;
        next_model_sender.send(model.clone());
        let (count_new, count_updates) = patch.is_new.iter().fold(
            (0, 0), |(new, old), t| { if *t { (new + 1, old) } else { (new, old + 1) } });
        debug!("model_manager, new updates, {}, {}, {}, {}, {}, {}, {}, {}",
                machine_id, node_id, node_count, model.depth[node_id], remote_gamma, patch.size,
                count_new, count_updates);
        if upload_model(&model, &model_sig, gamma, root_gamma, exp_name) {
            debug!("model_manager, accept, {}, {}, {}, {}",
                    old_sig, model_sig, machine_name, patch.size);
        } else {
            debug!("model_manager, upload failed, {}, {}, {}, {}",
                    old_sig, model_sig, machine_name, patch.size);
        }
        last_timestamp = global_timer.get_duration();
        write_model(&model, last_timestamp, true);
        for depth in new_nodes_depth {
            node_status.push(((depth, 0), 1.0, None));
            node_sum_gamma_sq.push(0.0);
            node_timestamp.push(0.0);
            avail_nodes += 1;
        }
        node_sum_gamma_sq[node_id] += remote_gamma * remote_gamma * patch.size as f32;
        let ((depth, _), gamma, machine_id) = node_status[node_id];
        node_status[node_id] =
            ((depth, model.children[node_id].len()), gamma, machine_id);
        if node_id == 0 && model.children[node_id].len() >= max_num_trees ||
                node_id > 0 && model.children[node_id].len() > 1 {
            avail_nodes -= 1;
            node_status[node_id].2 = None;
        }
    }
    info!("Model sync quits, {}, {}, {}, {}, Model length: {}, Is gamma significant? {}",
            state, gamma >= min_gamma, num_iterations <= 0, model.size() < num_iterations,
            model.size(), gamma >= min_gamma);
    write_model(&model, last_timestamp, false);
    {
        debug!("sampler state, false, model sync quits");
        let mut state = sampler_state.write().unwrap();
        *state = false;
    }
}


fn update_assignments(
    node_status: &mut Vec<((usize, usize), f32, Option<usize>)>,
    worker_assign: &mut Vec<Option<usize>>,
    gamma: f32,
    root_gamma: f32,
    max_depth: usize,
    max_num_trees: usize,
    exp_name: &String,
    _parents: &Vec<usize>,
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
            let mut node = None;
            let mut status_log = (0, 0);
            for i in 0..node_status.len() {
                let ((depth, num_child), old_gamma, status) = node_status[i];
                if depth < max_depth && status.is_none() &&
                        (i == 0 && num_child < max_num_trees || i > 0 && num_child <= 1) &&
                        (i > 0 && old_gamma > gamma || i == 0 && old_gamma > root_gamma) {
                    node = Some(i);
                    status_log = (depth, num_child);
                    break;
                }
            }
            if node.is_some() {
                let node = node.unwrap();
                let (status, gamma, _) = node_status[node];
                node_status[node] = (status, gamma, Some(valid_worker));
                node_sum_gamma_sq[node] = 0.0;
                node_timestamp[node] = cur_timestamp;
                worker_assign[valid_worker] = Some(node);
                did_something = true;
                num_updates += 1;
                debug!("model-manager, assign, {}, {}, {}, {}",
                       valid_worker, node, status_log.0, status_log.1);
            }
        }
    }
    if num_updates > 0 {
        debug!("model-manager, assign updates, {}", num_updates);
        upload_assignments(&worker_assign, &exp_name);
    }
}