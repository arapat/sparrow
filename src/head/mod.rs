/// Accept and maintain a global model
/// ![](/images/taskmanager.png)
pub mod model_manager;
/// Generate new samples
/// ![](/images/sampler.png)
pub mod sampler;
/// Assign tasks to the scanners
pub mod scheduler;
mod model_with_version;

use std::sync::mpsc;
use std::sync::Mutex;

use config::Config;
use config::SampleMode;
use commons::bins::Bins;
use commons::Model;

use commons::channel;
use commons::packet::TaskPacket;
use commons::packet::UpdatePacket;
use self::sampler::start_sampler_async;
use self::scheduler::Scheduler;
use self::model_manager::ModelManager;
use self::model_with_version::ModelWithVersion;

use tmsn::Network;


/// Start the components resided in head node
pub fn start_head(
    config: Config,
    sample_mode: SampleMode,
    bins: Vec<Bins>,
    init_tree: Model,
) {
    let (task_packet_sender, task_packet_receiver) = mpsc::channel();

    // Pass the models between the network to the Strata
    let (sampler_model_s, sampler_model_r) = channel::bounded(config.channel_size, "updated-models");

    let model = ModelWithVersion::new(init_tree, "Head".to_string());

    // 1. Start sampler
    // let sampler_state = start_sampler_async(
    // TODO: add a way to stop sampler
    start_sampler_async(
        &config,
        &sample_mode,
        &bins,
        &model.model,
        sampler_model_r,
        task_packet_sender.clone(),
    );

    // 2. Create a model manager
    let mut model_sync = ModelManager::new(&model);

    // 3. Create a scheduler
    let mut scheduler = Scheduler::new(config.network.len(), config.min_grid_size, &config);
    // let mut model = ModelWithVersion::new(init_tree.clone(), "Sampler".to_string());
    // scheduler.set_assignments(model, 1.0);

    let mutex_task_packet_sender = Mutex::new(task_packet_sender.clone());
    // TODO: increase capacity
    let capacity = 1;
    let mut network = Network::new(config.port, &config.network,
        Box::new(move |from_addr: String, _to_addr: String, update_packet: String| {
            let task_packet_sender = mutex_task_packet_sender.lock().unwrap();
            let mut packet: UpdatePacket = serde_json::from_str(&update_packet).unwrap();
            let mut model = model_sync.handle_packet(&from_addr, &mut packet);
            let (gamma, assigns) = scheduler.handle_packet(
                &from_addr, &mut packet, &mut model, capacity);

            // update the sampler
            sampler_model_s.send(model.model.clone());

            // update the scanners
            let mut task_packet = TaskPacket::new();
            task_packet.set_model(model.model);
            task_packet.set_gamma(gamma);
            assigns.into_iter().for_each(|(addr, task)| {
                task_packet.set_expand_node(Some(task));
                task_packet_sender.send((Some(addr), task_packet.clone())).unwrap();
            });
            // other scanners also receive an update without updating their tasks
            task_packet.set_expand_node(None);
            task_packet_sender.send((None, task_packet)).unwrap();
            drop(task_packet_sender);
        }),
        false,
    );

    // activate first scanner
    let mut task_packet = TaskPacket::new();
    task_packet.set_model(model.model);
    task_packet.set_gamma(config.default_gamma);
    task_packet.set_expand_node(Some(0));
    let first_scanner = network.get_subscribers()[0].clone();
    task_packet_sender.send((Some(first_scanner), task_packet.clone())).unwrap();

    // launch network send
    let mut _current_sample_version = 0;
    network.set_health_parameter(10);
    for (packet_id, (dest, mut task)) in task_packet_receiver.iter().enumerate() {
        task.set_packet_id(packet_id);
        if task.new_sample_version.is_none() {
            _current_sample_version = task.new_sample_version.as_ref().unwrap().clone();
        }

        let task_json = serde_json::to_string(&task).unwrap();
        info!("head packet, {}", task_json);
        network.send(dest, task_json).unwrap();
    }

    // TODO: when to stop?
    //     self.num_trees <= 0 || self.model.model.tree_size < self.num_trees
    info!("Head node quits.");
    // let final_model = write_model(&self.model.model, self.model_ts, false);
    // debug!("model_manager, final model, {}", final_model);
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