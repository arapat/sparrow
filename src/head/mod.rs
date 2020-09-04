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
use commons::model::Model;

use commons::channel;
use commons::packet::TaskPacket;
use commons::packet::UpdatePacket;
use commons::packet::UpdatePacketType;
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
    let starting_model_size = model.size();

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
    scheduler.set_assignments(&model, config.default_gamma, 1);

    let mutex_task_packet_sender = Mutex::new(task_packet_sender.clone());
    let mutex_packet = Mutex::new(TaskPacket::new());
    // TODO: increase capacity
    let capacity = 1;
    let mut network = Network::new(config.port, &config.network,
        Box::new(move |from_addr: String, update_packet: String| {
            let mut packet: UpdatePacket = serde_json::from_str(&update_packet).unwrap();
            debug!("received a packet, {}", packet.packet_id);
            packet.set_packet_type(model_sync.size());
            let mut model = model_sync.handle_packet(&from_addr, &mut packet);
            let (gamma, _assigns) = scheduler.handle_packet(
                &from_addr, &mut packet, &mut model, capacity);

            // update the sampler
            sampler_model_s.send(model.model.clone());

            // update the scanners
            let task_packet_sender = mutex_task_packet_sender.lock().unwrap();
            let mut task_packet = TaskPacket::new();
            task_packet.set_model(model.model);
            task_packet.set_gamma(gamma);
            // TODO: use expand node
            task_packet.set_expand_node(None);
            let mut curr_packet = mutex_packet.lock().unwrap();
            if *curr_packet != task_packet {
                task_packet_sender.send((None, task_packet.clone())).unwrap();
                *curr_packet = task_packet;
            } else if packet.packet_type == UpdatePacketType::Empty {
                task_packet.set_dest(&from_addr);
                task_packet_sender.send((Some(from_addr), task_packet.clone())).unwrap();
            }
            drop(curr_packet);
            drop(task_packet_sender);
        }),
        false,
    );

    // activate first scanner
    info!("Activate first scanner");
    let mut task_packet = TaskPacket::new();
    task_packet.set_model(model.model);
    task_packet.set_gamma(config.default_gamma);
    task_packet.set_expand_node(Some(0));
    let _first_scanner = network.get_subscribers()[0].clone();
    // task_packet_sender.send((Some(first_scanner), task_packet.clone())).unwrap();
    task_packet_sender.send((None, task_packet.clone())).unwrap();

    // launch network send
    let mut _current_sample_version = 0;
    network.set_health_parameter(10);
    for (packet_id, (dest, mut task)) in task_packet_receiver.iter().enumerate() {
        // Stop the head node when the goal is reached
        if task.model.is_some() && task.model.as_ref().unwrap().size() >=
                starting_model_size + config.num_trees {
            task.new_sample_version = None;
            task.model = None;
        }

        task.set_packet_id(packet_id);
        if task.new_sample_version.is_some() {
            _current_sample_version = task.new_sample_version.as_ref().unwrap().clone();
        }

        let task_json = serde_json::to_string(&task).unwrap();
        info!("head packet, {:?}, {}", dest, task_json);
        network.send(dest, task_json).unwrap();
        if task.new_sample_version.is_none() && task.model.is_none() {
            break;
        }
    }

    // TODO: when to stop?
    //     self.num_trees <= 0 || self.model.model.size() < self.num_trees
    info!("Head node quits.");
    // let final_model = write_model(&self.model.model, self.model_ts, false);
    // debug!("model_manager, final model, {}", final_model);
}
