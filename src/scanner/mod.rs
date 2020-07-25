/// The implementation of the AdaBoost algorithm with early stopping rule.
pub mod booster;
/// A data loader with two independent caches. Alternatively, we use one
/// of the caches to feed data to the boosting algorithm, and the other
/// to load next sample set.
pub mod buffer_loader;

use head::model_manager::gamma::Gamma;
use commons::Model;
use commons::bins::Bins;
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


#[derive(PartialEq)]
pub enum BoosterState {
    IDLE,
    STOPPING,
    RUNNING,
}


#[derive(Serialize, Deserialize, Debug)]
struct Packet {
    pub packet_id: usize,
    pub model: Option<Model>,
    pub gamma: Option<Gamma>,
    pub assignment: usize,
    pub new_sample_version: Option<usize>,
}


pub fn start(config: Config, sample_mode: SampleMode, bins: Vec<Bins>, init_tree: Model) {
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
    let booster_state = Arc::new(RwLock::new(BoosterState::IDLE));
    let sampler_signal_sender = Mutex::new(sampler_signal_sender);
    let buffer_loader_m = Mutex::new(Some(buffer_loader));
    let mut network = Network::new(config.port, &vec![], Box::new(move |packet: Packet| {
        debug!("Received a new packet from head");
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
                config.exp_name.clone(),
                init_tree.clone(),
                config.num_trees,
                config.num_splits,
                config.num_features,
                config.min_gamma,
                training_loader,
                bins.clone(),
                config.max_sample_size,
                config.default_gamma,
                config.save_process,
                config.save_interval,
                ro_booster_state,
            );
            // TODO: not necessary?
            // booster.enable_network(config.local_name.clone(), config.port);
            debug!("Booster ready to train");
            booster.training();

            // Release the buffer loader
            let mut buffer_loader = buffer_loader_m.lock().unwrap();
            *buffer_loader = Some(booster.training_loader);
            // Mark booster as idle
            let mut booster_state = booster_state.write().unwrap();
            *booster_state = BoosterState::IDLE;
            drop(buffer_loader);
            drop(booster_state);
        }
    }));
    network.set_health_parameter(10);
}