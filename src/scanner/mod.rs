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
use commons::Model;
use config::Config;
use config::SampleMode;

use self::booster::Boosting;
use self::buffer_loader::BufferLoader;

use std::sync::mpsc;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::Mutex;
use std::thread::sleep;
use std::time::Duration;

use tmsn::Network;


pub fn start_scanner(config: Config, sample_mode: SampleMode, bins: Vec<Bins>) {
    println!("Starting scanner");

    debug!("Starting the buffered loader.");
    // sending signals to the sample loader
    let (sampler_signal_sender, sampler_signal_receiver) = mpsc::channel();
    sampler_signal_sender.send(0).unwrap();
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
    let (new_updates_sender, new_updates_receiver) = mpsc::channel();
    let new_updates_sender = Mutex::new(new_updates_sender);
    let packet_id = Mutex::new(0);
    let booster_state = Arc::new(RwLock::new(BoosterState::IDLE));
    let sampler_signal_sender = Mutex::new(sampler_signal_sender);
    let buffer_loader_m = Mutex::new(Some(buffer_loader));
    let mut network = Network::new(config.port, &vec![],
        Box::new(move |from_addr: String, to_addr: String, task_packet: String| {
            debug!("Received a new packet from head, {}, {}", from_addr, to_addr);
            let packet: TaskPacket = serde_json::from_str(&task_packet).unwrap();
            if packet.new_sample_version.is_some() {
                debug!("Packet is a new sample signal");
                let sampler_signal_sender = sampler_signal_sender.lock().unwrap();
                let new_version = packet.new_sample_version.as_ref().unwrap().clone();
                sampler_signal_sender.send(new_version).unwrap();
                drop(sampler_signal_sender);
            } else {
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
                let mut buffer_loader = buffer_loader_m.lock().unwrap();
                let training_loader = buffer_loader.take().unwrap();
                drop(buffer_loader);
                let ro_booster_state = booster_state.clone();
                let mut booster = Boosting::new(
                    packet,
                    ro_booster_state,
                    training_loader,
                    bins.clone(),
                    &config,
                );
                debug!("Booster ready to train");
                booster.training();

                let (prev_packet, model, mut buffer_loader) = booster.destroy();
                // send out updates
                let mut packet_id = packet_id.lock().unwrap();
                let updates_packet = get_packet(*packet_id, &model, prev_packet, &buffer_loader);
                let updates_json = serde_json::to_string(&updates_packet).unwrap();
                *packet_id += 1;
                drop(packet_id);
                let new_updates_sender = new_updates_sender.lock().unwrap();
                new_updates_sender.send(updates_json).unwrap();
                drop(new_updates_sender);
                // reset buffer scores
                buffer_loader.reset_scores();
                let mut b = buffer_loader_m.lock().unwrap();
                *b = Some(buffer_loader);
                drop(b);
                // mark booster as idle
                let mut booster_state = booster_state.write().unwrap();
                *booster_state = BoosterState::IDLE;
                drop(booster_state);
            }
        }),
        false,
    );
    network.set_health_parameter(10);
    for (_packet_id, new_updates) in new_updates_receiver.iter().enumerate() {
        network.send(new_updates).unwrap();
    }
}


fn get_packet(
    packet_id: usize, model: &Model, task: TaskPacket, buffer_loader: &BufferLoader,
) -> UpdatePacket {
    let base_model_size = task.model.as_ref().unwrap().size();
    let tree_slice = model.model_updates.create_slice(base_model_size..model.size());
    UpdatePacket::new(
        packet_id,
        tree_slice,
        task,
        buffer_loader.current_version,
        buffer_loader.ess,
    )
}