/// The implementation of the AdaBoost algorithm with early stopping rule.
pub mod booster;
/// A data loader with two independent caches. Alternatively, we use one
/// of the caches to feed data to the boosting algorithm, and the other
/// to load next sample set.
pub mod buffer_loader;

use commons::bins::Bins;
use commons::packet::BoosterState;
use commons::packet::TaskPacket;
use commons::packet::UpdatePacket;
use commons::model::Model;
use config::Config;
use config::SampleMode;

use self::booster::Boosting;
use self::buffer_loader::BufferLoader;

use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::Mutex;
use std::thread::sleep;
use std::thread::spawn;
use std::time::Duration;

use tmsn::Network;


pub fn start_scanner(
    config: Config, sample_mode: SampleMode, bins: Vec<Bins>,
) -> (Network, Receiver<UpdatePacket>) {
    println!("Starting scanner");

    debug!("Starting the buffered loader.");
    // sending signals to the sample loader
    let (sampler_signal_sender, sampler_signal_receiver) = mpsc::channel();
    let buffer_loader = BufferLoader::new(
        config.buffer_size,
        config.batch_size,
        sample_mode.clone(),
        config.sleep_duration,
        config.min_ess,
        config.exp_name.clone(),
        sampler_signal_receiver,
    );

    debug!("Starting the network.");
    // sending signals to the scanner
    let debug_mode = config.debug_mode;
    let (new_updates_sender, new_updates_receiver) = mpsc::channel();
    let new_updates_sender = Arc::new(Mutex::new(new_updates_sender));
    let booster_state = Arc::new(RwLock::new(BoosterState::IDLE));
    let sampler_signal_sender = Mutex::new(sampler_signal_sender);
    let buffer_loader = Arc::new(Mutex::new(Some(buffer_loader)));
    let mut curr_packet: Option<TaskPacket> = None;

    let network = Network::new(config.port, &vec![],
        Box::new(move |from_addr: String, to_addr: String, task_packet: String| {
            debug!("Received a new packet from head, {}, {}, {}", from_addr, to_addr, task_packet);
            let packet: TaskPacket = serde_json::from_str(&task_packet).unwrap();
            if packet.new_sample_version.is_some() {
                debug!("Packet is a new sample signal");
                let sampler_signal_sender = sampler_signal_sender.lock().unwrap();
                let new_version = packet.new_sample_version.as_ref().unwrap().clone();
                sampler_signal_sender.send(new_version).unwrap();
                drop(sampler_signal_sender);
            } else if curr_packet.is_none() && packet.expand_node.is_some() ||
                      !curr_packet.as_ref().unwrap().equals(&packet) {
                curr_packet = Some(packet.clone_with_expand(&curr_packet));
                let (packet, booster_state, buffer_loader, new_updates_sender, bins, config) =
                    (curr_packet.as_ref().unwrap().clone(), booster_state.clone(),
                     buffer_loader.clone(), new_updates_sender.clone(), bins.clone(),
                     config.clone());
                spawn(move || start_booster(
                    packet, booster_state, buffer_loader, new_updates_sender, bins, config));
            } else {
                info!("Package is ignored, {}", packet.packet_id);
            }
        }),
        debug_mode,
    );

    (network, new_updates_receiver)
}

fn start_booster(
    packet: TaskPacket,
    booster_state: Arc<RwLock<BoosterState>>,
    buffer_loader: Arc<Mutex<Option<BufferLoader>>>,
    new_updates_sender: Arc<Mutex<mpsc::Sender<UpdatePacket>>>,
    bins: Vec<Bins>,
    config: Config,
) {
    debug!("Stopping existing booster");
    let mut is_booster_stopped = false;
    while !is_booster_stopped {
        let mut booster_state = booster_state.write().unwrap();
        is_booster_stopped = *booster_state == BoosterState::IDLE;
        if !is_booster_stopped {
            *booster_state = BoosterState::STOPPING;
        }
        drop(booster_state);
        sleep(Duration::from_millis(500));
    }

    debug!("Starting the booster.");
    let mut read_loader = buffer_loader.lock().unwrap();
    let training_loader = read_loader.take().unwrap();
    drop(read_loader);
    let mut w_booster_state = booster_state.write().unwrap();
    *w_booster_state = BoosterState::RUNNING;
    drop(w_booster_state);
    let ro_booster_state = booster_state.clone();
    let mut booster = Boosting::new(
        packet,
        ro_booster_state,
        training_loader,
        bins,
        config,
    );
    debug!("Booster ready to train");
    booster.training();

    let (prev_packet, model, mut loader) = booster.destroy();
    // send out updates
    let updates_packet = get_packet(&model, prev_packet, &loader);
    let new_updates_sender = new_updates_sender.lock().unwrap();
    new_updates_sender.send(updates_packet).unwrap();
    drop(new_updates_sender);
    // reset buffer scores
    loader.reset_scores();
    let mut write_loader = buffer_loader.lock().unwrap();
    *write_loader = Some(loader);
    drop(write_loader);
    // mark booster as idle
    let mut booster_state = booster_state.write().unwrap();
    *booster_state = BoosterState::IDLE;
    drop(booster_state);
}

/// Sending out the messages generated by the scanner over tmsn, this function will block.
pub fn handle_network_send(network: &mut Network, new_updates_receiver: Receiver<UpdatePacket>) {
    network.set_health_parameter(10);
    for (packet_id, mut new_updates) in new_updates_receiver.iter().enumerate() {
        new_updates.set_packet_id(packet_id);
        let updates_json = serde_json::to_string(&new_updates).unwrap();
        info!("scanner packet, {}", updates_json);
        network.send(None, updates_json).unwrap();
    }
}


fn get_packet(model: &Model, task: TaskPacket, buffer_loader: &BufferLoader) -> UpdatePacket {
    let tree = model.get_last_tree();
    UpdatePacket::new(tree, task, buffer_loader.current_version, buffer_loader.ess)
}


#[cfg(test)]
mod test {
    use prep_training;
    use scanner::start_scanner;
    use std::io::Write;
    use time::get_time;

    #[test]
    fn test_scanner() {
        init_env_logger();

        let config_path = "./examples/config_splice_debug.yaml".to_string();
        let (config, sample_mode, bins) = prep_training(&config_path);
        let (mut network, new_updates_receiver) = start_scanner(config, sample_mode, bins);

        let source = "source".to_string();
        let target = "target".to_string();
        let packet1 = r#"{"packet_id":0,"model":{"tree_size":0,"parent":[],"children":[],"split_feature":[],"threshold":[],"evaluation":[],"predicts":[],"is_active":[],"depth":[],"base_version":0,"model_updates":{"size":0,"parent":[],"feature":[],"threshold":[],"evaluation":[],"predicts":[],"condition":[],"is_new":[]}},"gamma":0.25,"expand_node":0,"new_sample_version":null}"#;
        let packet2 = r#"{"packet_id":1,"model":null,"gamma":null,"expand_node":null,"new_sample_version":1}"#;
        let packet3 = r#"{"packet_id":2,"model":{"tree_size":1,"parent":[0],"children":[[]],"split_feature":[0],"threshold":[0],"evaluation":[false],"predicts":[-2.9748201],"is_active":[],"depth":[0],"base_version":1,"model_updates":{"size":1,"parent":[-1],"feature":[0],"threshold":[0],"evaluation":[false],"predicts":[-2.9748201],"condition":[[]],"is_new":[true]}},"gamma":0.25,"expand_node":null,"new_sample_version":null}"#;
        let packet4 = r#"{"packet_id":3,"model":{"tree_size":3,"parent":[0,0,0],"children":[[1,2],[],[]],"split_feature":[0,329,329],"threshold":[0,0,0],"evaluation":[false,true,false],"predicts":[-2.886624,-1.0294012,1.0294012],"is_active":[],"depth":[0,1,1],"base_version":3,"model_updates":{"size":3,"parent":[-1,0,0],"feature":[0,329,329],"threshold":[0,0,0],"evaluation":[false,true,false],"predicts":[-2.886624,-1.0294012,1.0294012],"condition":[[],[[329,0,true]],[[329,0,false]]],"is_new":[true,true,true]}},"gamma":0.25,"expand_node":null,"new_sample_version":null}"#;
        let packs = vec![packet3, packet4];

        network.mock_send(&source, &target, Some(packet1.to_string()));
        network.mock_send(&source, &target, Some(packet2.to_string()));

        for (packet_id, new_updates) in new_updates_receiver.iter().enumerate() {
            println!("debug scanner, {}, {:?}", packet_id, new_updates);
            if packet_id >= packs.len() {
                break;
            }
            network.mock_send(&source, &target, Some(packs[packet_id].to_string()));
        }
    }

    fn init_env_logger() {
        let curr_time = get_time().sec;
        env_logger::Builder::from_default_env()
            .format(move |buf, record| {
                let timestamp = get_time();
                let formatted_ts = format!("{}.{}", timestamp.sec - curr_time, timestamp.nsec);
                writeln!(
                    buf, "{}, {}, {}, {}",
                    record.level(), formatted_ts, record.module_path().unwrap(), record.args()
                )
            })
            .init();
    }
}