use std::sync::mpsc;
use std::thread::spawn;
use bincode::serialize;
use bincode::deserialize;
use commons::Model;
use commons::ModelSig;
use commons::channel::bounded;
use commons::channel::Sender;
use commons::channel::Receiver;
use commons::io::load_s3 as io_load_s3;
use commons::io::write_s3 as io_write_s3;
use commons::io::delete_s3;
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
    // TODO: make duration a config parameter
    const DURATION: f32 = 5.0;
    let mut model_sig = "".to_string();
    let mut model = Tree::new(1, 0.0, 0.0);
    // let mut gamma = HashMap::new();
    let mut timer = PerformanceMonitor::new();
    let mut gamma_max = default_gamma;
    let mut alpha = 0.5;
    timer.start();
    loop {
        let packet = receiver.try_recv();
        if packet.is_none() {
            if timer.get_duration() >= DURATION {
                gamma_max = gamma_max * 0.9;
                alpha = 0.5;
                let current_gamma = gamma_max - gamma_max * alpha * 0.1;
                if upload_model(&model, &model_sig, current_gamma) {
                    debug!("model_manager, gamma updated, {}, {}, {}", gamma_max, alpha, current_gamma);
                } else {
                    debug!("model_manager, gamma update failed, {}, {}, {}",
                        gamma_max, alpha, current_gamma);
                }
                timer.reset();
                timer.start();
            }
            continue;
        }
        if sender.is_full() {
            alpha = alpha / 2.0;
        }
        let current_gamma = gamma_max - gamma_max * alpha * 0.1;
        let (patch, remote_gamma, old_sig, new_sig) = packet.unwrap();
        let machine_name: String = (*old_sig).split("_").next()
                                        .expect("Invalid signature format")
                                        .to_string();
        // *(gamma.entry(machine_name).or_insert(0.0)) = remote_gamma;
        if old_sig != model_sig {
            debug!("model_manager, reject for base model mismatch, {}, {}", model_sig, old_sig);
            continue;
        }
        // let current_gamma: f32 = gamma.values().fold(0.0, |a, &b| a.max(b));
        /*
        if gamma.len() < num_machines {
            debug!("model_manager, reject for scanners not ready, {}, {}",
                   gamma.len(), num_machines);
            continue;
        }
        if current_gamma + 1e-8 < remote_gamma {
            debug!("model_manager, reject for small gamma, {}, {}", current_gamma, remote_gamma);
            continue;
        }
        */
        model.append_patch(&patch, remote_gamma, old_sig == "");
        model_sig = new_sig;
        next_model_sender.send(model.clone());
        if upload_model(&model, &model_sig, current_gamma) {
            debug!("model_manager, accept, {}, {}", old_sig, model_sig);
        } else {
            debug!("model_manager, upload failed, {}, {}", old_sig, model_sig);
        }
    }
}


pub fn clear_s3() {
    delete_s3(REGION, BUCKET, S3_PATH, FILENAME);
}
