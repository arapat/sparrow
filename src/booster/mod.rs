mod learner;

use std::fs::File;
use std::io::BufWriter;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::sync::mpsc;
use std::thread::sleep;
use std::time::Duration;
use serde_json;
use tmsn::network::start_network_only_send;

use buffer_loader::BufferLoader;
use commons::is_zero;
use commons::io::create_bufwriter;
use commons::Model;
use commons::performance_monitor::PerformanceMonitor;
use commons::ModelSig;
use commons::bins::Bins;
use model_sync::download_assignments;
use model_sync::download_model;
use self::learner::get_base_node;
use self::learner::Learner;


/// The boosting algorithm. It contains two functions, one for starting
/// the network communication, the other for starting the training procedure.
pub struct Boosting {
    exp_name: String,
    num_iterations: usize,
    training_loader: BufferLoader,

    learner: Learner,
    model: Model,
    base_model_sig: String,
    base_model_size: usize,
    last_sent_model_sig: String,
    last_sent_sample_version: usize,
    last_expand_node: usize,
    last_sent_gamma: f32,

    network_sender: Option<mpsc::Sender<ModelSig>>,
    local_name: String,
    local_id: usize,
    last_remote_length: usize,

    persist_id: u32,
    persist_file_buffer: Option<BufWriter<File>>,
    save_interval: usize,

    max_sample_size: usize,
    save_process: bool,
}

impl Boosting {
    /// Create a boosting training class.
    ///
    /// * `num_iterations`: the number of boosting iteration. If it equals to 0, then the algorithm runs indefinitely.
    /// * `training_loader`: the double-buffered data loader that provides examples to the algorithm.
    /// over multiple workers, it might be a subset of the full feature set.
    /// * `max_sample_size`: the number of examples to scan for determining the percentiles for the features.
    /// * `default_gamma`: the initial value of the edge `gamma` of the candidate valid weak rules.
    pub fn new(
        exp_name: String,
        init_model: Model,
        num_iterations: usize,
        num_features: usize,
        min_gamma: f32,
        training_loader: BufferLoader,
        // serial_training_loader: SerialStorage,
        bins: Vec<Bins>,
        max_sample_size: usize,
        default_gamma: f32,
        save_process: bool,
        save_interval: usize,
    ) -> Boosting {
        // TODO: make num_cadid a paramter
        let learner = Learner::new(
            min_gamma, default_gamma, num_features, bins);

        let persist_file_buffer = {
            if save_process {
                None
            } else {
                Some(create_bufwriter(&String::from("model.json")))
            }
        };
        Boosting {
            exp_name: exp_name,
            num_iterations: num_iterations,
            training_loader: training_loader,

            learner: learner,
            model: init_model,
            base_model_sig: "".to_string(),
            base_model_size: 0,
            last_sent_model_sig: ".".to_string(),
            last_sent_sample_version: 0,
            last_expand_node: 0,
            last_sent_gamma: 1.0,

            network_sender: None,
            local_name: "".to_string(),
            local_id: 0,
            last_remote_length: 0,

            persist_id: 0,
            persist_file_buffer: persist_file_buffer,
            save_interval: save_interval,

            max_sample_size: max_sample_size,
            save_process: save_process,
        }
    }


    pub fn set_root_tree(&mut self) {
        while !self.training_loader.try_switch() {
            sleep(Duration::from_millis(2000));
        }
        let max_sample_size = self.max_sample_size;
        let (_, base_pred, base_gamma) = get_base_node(max_sample_size, &mut self.training_loader);
        let index = self.model.add_node(-1, 0, 0, false, base_pred, base_gamma);
        let index = {
            if index.is_some() {
                index.unwrap()
            } else {
                0
            }
        };
        self.update_model();
        info!("scanner, added new rule, {}, {}, {}, {}",
              self.model.size(), max_sample_size, max_sample_size, index);
    }

    fn update_model(&mut self) {
        let model = self.training_loader.new_model.read().unwrap();
        if model.is_some() {
            self.model = model.as_ref().unwrap().clone();
        }
    }

    /// Enable network communication. `name` is the name of this worker, which can be arbitrary
    /// and is only used for debugging purpose.
    /// `port` is the port number that used for network communication.
    pub fn enable_network(&mut self, name: String, port: u16) {
        let (local_s, local_r): (mpsc::Sender<ModelSig>, mpsc::Receiver<ModelSig>) =
            mpsc::channel();
        start_network_only_send(name.as_ref(), port, local_r);
        // let (hb_s, hb_r): (mpsc::Sender<String>, mpsc::Receiver<String>) = mpsc::channel();
        // start_network_only_send(name.as_ref(), port + 1, hb_r);
        self.network_sender = Some(local_s);
        self.local_name = name.clone();
        self.local_id = {
            let t: Vec<&str> = self.local_name.rsplitn(2, '_').collect();
            t[0].parse().unwrap()
        };
        // heartbeat
        /*
        {
            loop {
                hb_s.send(name.clone()).unwrap();
                sleep(Duration::from_secs(5));
            }
        }
        */
    }


    /// Start training the boosting algorithm.
    pub fn training(
        &mut self,
        prep_time: f32,
    ) {
        debug!("Start training.");

        while self.base_model_sig != "init" {
            self.handle_network(false);
            sleep(Duration::from_secs(2));
        }
        debug!("booster, remote model is downloaded");

        let init_sampling_duration = self.training_loader.get_sampling_duration();
        let mut global_timer = PerformanceMonitor::new();
        let mut learner_timer = PerformanceMonitor::new();
        global_timer.start();

        let mut iteration = 0;
        let mut is_gamma_significant = true;
        let mut total_data_size = 0;
        while is_gamma_significant &&
                (self.num_iterations <= 0 || self.model.size() < self.num_iterations) {
            let (new_rule, batch_size, switched) = {
                let (data, switched) =
                    self.training_loader.get_next_batch_and_update(true, &self.model);
                learner_timer.resume();
                if switched {
                    self.learner.reset();
                }
                (self.learner.update(&self.model, &data), data.len(), switched)
            };
            if switched {
                self.update_model();
            }
            learner_timer.update(batch_size);
            global_timer.update(batch_size);
            learner_timer.pause();
            total_data_size += batch_size;

            if new_rule.is_some() {
                let new_rule = new_rule.unwrap();
                new_rule.write_log();
                let index = self.model.add_node(
                    new_rule.prt_index as i32,
                    new_rule.feature,
                    new_rule.threshold,
                    new_rule.evaluation,
                    new_rule.predict,
                    new_rule.gamma,
                );
                let index = {
                    if index.is_some() {
                        index.unwrap()
                    } else {
                        0
                    }
                };
                info!("scanner, added new rule, {}, {}, {}, {}",
                      self.model.size(), new_rule.num_scanned, total_data_size, index);

                // post updates
                is_gamma_significant = self.learner.is_gamma_significant();
                self.learner.reset();
                if self.model.size() % self.save_interval == 0 {
                    self.handle_persistent(iteration, prep_time + global_timer.get_duration());
                }
                total_data_size = 0;
            }

            iteration += 1;
            let data_size = self.training_loader.size;
            if self.handle_network(total_data_size >= data_size) {
                total_data_size = 0;
            }

            let sampling_duration = self.training_loader.get_sampling_duration() - init_sampling_duration;
            global_timer.set_adjust(-sampling_duration);
            global_timer.write_log("boosting-overall");
            learner_timer.write_log("boosting-learning");
        }
        self.handle_persistent(iteration, prep_time + global_timer.get_duration());
        info!("Training is finished. Model length: {}. Is gamma significant? {}.",
              self.model.size(), self.learner.is_gamma_significant());
    }

    fn handle_network(&mut self, is_full_scanned: bool) -> bool {
        if self.network_sender.is_none() {
            return false;
        }

        let mut is_packet_sent = false;
        // 0. Get the latest model
        // 1. If it is newer, overwrite local model
        // 2. Otherwise, push the current update to remote
        let model_score = download_model(&self.exp_name);
        if model_score.is_some() {
            let (remote_model, remote_model_sig, current_gamma, root_gamma): (Model, String, f32, f32) =
                model_score.unwrap();
            let new_model_sig = self.local_name.clone() + "_" + &self.model.size().to_string();
            let has_new_node = self.model.size() > self.base_model_size && (
                self.last_sent_model_sig != new_model_sig ||
                self.training_loader.current_version != self.last_sent_sample_version
            );
            let has_empty_message = is_full_scanned && (
                self.last_expand_node != self.learner.expand_node ||
                !is_zero(self.last_sent_gamma - self.learner.rho_gamma)
            );
            {
                debug!("debug-empty, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                       has_new_node, self.model.size(), self.base_model_size,
                       self.last_sent_model_sig, new_model_sig,
                       self.training_loader.current_version, self.last_sent_sample_version,
                       has_empty_message, is_full_scanned, self.last_expand_node,
                       self.learner.expand_node, self.last_sent_gamma, self.learner.rho_gamma);
            };
            if remote_model_sig != self.base_model_sig {
                // replace the existing model
                let old_size = self.model.size();
                self.model = remote_model;
                self.base_model_sig = remote_model_sig;
                self.base_model_size = self.model.size();
                self.last_remote_length = self.model.size();
                self.learner.reset();
                debug!("model-replaced, {}, {}, {}",
                       self.model.size(), old_size, self.base_model_sig);
            } else if has_new_node || has_empty_message {
                // send out the local patch
                let tree_slice = self.model.model_updates.create_slice(
                    self.last_remote_length..self.model.size());
                let packet: ModelSig = (
                    tree_slice, self.learner.rho_gamma, self.training_loader.current_version,
                    self.base_model_sig.clone(), new_model_sig.clone());
                let send_result = self.network_sender.as_ref().unwrap()
                                        .send(packet);
                if let Err(err) = send_result {
                    error!("Attempt to send the local model to the network module but failed.
                            Error: {}", err);
                } else {
                    self.last_sent_model_sig = new_model_sig;
                    self.last_sent_sample_version = self.training_loader.current_version;
                    if !has_new_node {
                        self.last_expand_node = self.learner.expand_node;
                        self.last_sent_gamma = self.learner.rho_gamma;
                    }
                    is_packet_sent = true;
                    info!("Sent the local model to the network module, {}, {}, {}, {}",
                        self.last_sent_model_sig, self.last_sent_sample_version,
                        self.last_remote_length, self.model.size());
                }
            }
            self.learner.set_gamma(current_gamma, root_gamma);
        } else {
            debug!("booster, download-model, failed");
        }

        let assigns = download_assignments(&self.exp_name);
        if assigns.is_some() {
            let assignments = assigns.unwrap();
            let expand_node = assignments[self.local_id % assignments.len()];
            if expand_node.is_some() {
                if self.learner.set_expand_node(expand_node.unwrap()) {
                    self.learner.reset();
                }
            }
        }
        is_packet_sent
    }

    fn handle_persistent(&mut self, iteration: usize, timestamp: f32) {
        let json = serde_json::to_string(&(timestamp, iteration, &self.model)).expect(
            "Local model cannot be serialized."
        );
        self.persist_id += 1;
        if self.save_process {
            let mut file_buffer = create_bufwriter(
                &format!("models/model_{}-v{}.json", self.model.size(), self.persist_id));
            file_buffer.write(json.as_ref()).unwrap();
        } else {
            let buf = self.persist_file_buffer.as_mut().unwrap();
            buf.seek(SeekFrom::Start(0)).unwrap();
            buf.write(json.as_ref()).unwrap();
        }
        {
            let mut file_buffer = create_bufwriter(&"model.json".to_string());
            file_buffer.write(json.as_ref()).unwrap();
        }
    }
}
