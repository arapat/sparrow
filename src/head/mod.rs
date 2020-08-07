/// Accept and maintain a global model
/// ![](/images/taskmanager.png)
pub mod model_manager;
/// Generate new samples
/// ![](/images/sampler.png)
pub mod sampler;
/// Assign tasks to the scanners
pub mod scheduler;


use std::path::Path;
use std::sync::mpsc;
use std::thread::sleep;
use std::time::Duration;

use config::Config;
use config::SampleMode;
use commons::bins::Bins;
use commons::Model;

use commons::channel;
use commons::io::raw_read_all;
use commons::packet::TaskPacket;
use self::sampler::start_sampler_async;
use self::model_manager::start_model_manager_async;

use tmsn::Network;


/// Start the components resided in head node
pub fn start_head(
    config: Config,
    sample_mode: SampleMode,
    bins: Vec<Bins>,
    init_tree: Model,
) {
    // TODO: add a way to stop sampler

    // Pass the models between the network to the Strata
    let (next_model_s, next_model_r) = channel::bounded(config.channel_size, "updated-models");
    start_model_manager_async(
        &config,
        &init_tree,
        &bins,
        next_model_s,
    );

    let (task_packet_sender, task_packet_receiver) = mpsc::channel();
    // with new network
    let mut network = Network::new(config.port, &vec![],
        Box::new(move |from_addr: String, to_addr: String, task_packet: String| {
        }),
        false,
    );
    // let sampler_state = start_sampler_async(
    start_sampler_async(
        &config,
        &sample_mode,
        &bins,
        &init_tree,
        next_model_r,
        task_packet_sender.clone(),
    );

    // launch network send
    let mut current_packet = TaskPacket::new();
    let mut _current_sample_version = 0;
    network.set_health_parameter(10);
    for (packet_id, mut task) in task_packet_receiver.iter().enumerate() {
        task.set_packet_id(packet_id);
        if task.new_sample_version.is_none() {
            _current_sample_version = task.new_sample_version.as_ref().unwrap().clone();
        } else {
            task.fill_none(&current_packet);
            current_packet = task.clone();
        }

        let task_json = serde_json::to_string(&task).unwrap();
        network.send(task_json).unwrap();
    }
}

/*
// Monitor running state, exit if state is false
loop {
    // Check if termination is manually requested
    let filename = "status.txt".to_string();
    if Path::new(&filename).exists() && raw_read_all(&filename).trim() == "0".to_string() {
        debug!("sampler state, false, change in the status.txt has been detected");
        *(sampler_state.write().unwrap()) = false;
        break;
    }
    sleep(Duration::from_secs(20));
}
debug!("State has been set to false. Main process to exit in 120 seconds.");
sleep(Duration::from_secs(120));
if std::fs::remove_file("status.txt").is_ok() {
    debug!("removed `status.txt`");
}
*/