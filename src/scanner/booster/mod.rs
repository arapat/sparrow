pub mod learner;
pub mod learner_helpers;

use std::sync::mpsc;
use std::thread::sleep;
use std::time::Duration;
use std::fmt::Display;
// TODO: use tmsn::network::start_network_only_send;
use commons::test_helper::start_network_only_send;

use commons::persistent_io::download_assignments;
use commons::persistent_io::download_model;
use commons::persistent_io::write_model;
use self::learner_helpers::get_base_node;

use commons::Model;
use scanner::buffer_loader::BufferLoader;
use commons::bins::Bins;
use commons::packet::Packet;
use commons::performance_monitor::PerformanceMonitor;
use commons::persistent_io::ModelPack;
use self::learner::Learner;
use self::learner::TreeNode;


pub const MODEL_SIG_PLACEHOLDER: &str = "MODEL_SIG_PLACEHOLDER";


/// The boosting algorithm. It contains two functions, one for starting
/// the network communication, the other for starting the training procedure.
pub struct Boosting {
    exp_name: String,
    num_trees: usize,
    training_loader: BufferLoader,

    learner: Learner,
    model: Model,
    base_model_sig: String,
    base_model_size: usize,

    network_sender: Option<mpsc::Sender<Packet>>,
    local_name: String,
    local_id: usize,
    packet_counter: usize,

    last_sent_model_length: usize,
    // for re-sending the non-empty packet if the sampler status changed
    // track: sample version
    is_sample_version_changed: bool,
    // for re-sending the empty packet if the scanner status changed
    // track: expanding node, gamma value
    is_scanner_status_changed: bool,
    // assignment is none if this scanner didn't receive any task from the sampler
    is_assignment_none: bool,

    persist_id: u32,
    save_interval: usize,

    max_sample_size: usize,
    save_process: bool,
    verbose: bool,
}

impl Boosting {
    /// Create a boosting training class.
    ///
    /// * `num_trees`: the number of boosting iteration. If it equals to 0, then the algorithm runs indefinitely.
    /// * `training_loader`: the double-buffered data loader that provides examples to the algorithm.
    /// over multiple workers, it might be a subset of the full feature set.
    /// * `max_sample_size`: the number of examples to scan for determining the percentiles for the features.
    /// * `default_gamma`: the initial value of the edge `gamma` of the candidate valid weak rules.
    pub fn new(
        exp_name: String,
        init_model: Model,
        num_trees: usize,
        num_splits: usize,
        num_features: usize,
        min_gamma: f32,
        training_loader: BufferLoader,
        bins: Vec<Bins>,
        max_sample_size: usize,
        default_gamma: f32,
        save_process: bool,
        save_interval: usize,
    ) -> Boosting {
        // TODO: make num_cadid a paramter
        let learner = Learner::new(
            min_gamma, default_gamma, num_features, bins, num_splits);
        Boosting {
            exp_name: exp_name,
            num_trees: num_trees,
            training_loader: training_loader,

            learner: learner,
            model: init_model,
            base_model_sig: "".to_string(),
            base_model_size: 0,

            is_sample_version_changed: true,
            is_scanner_status_changed: true,
            last_sent_model_length: 0,
            is_assignment_none: true,

            network_sender: None,
            local_name: "".to_string(),
            local_id: 0,
            packet_counter: 0,

            persist_id: 0,
            save_interval: save_interval,

            max_sample_size: max_sample_size,
            save_process: save_process,
            verbose: false,
        }
    }

    fn init(&mut self) {
        let mut timer = PerformanceMonitor::new();
        let mut last_reported_time = 0.0;
        timer.start();
        while self.training_loader.is_empty() {
            if timer.get_duration() - last_reported_time > 60.0 {
                last_reported_time = timer.get_duration();
                debug!("scanner, init, waiting the first sample, {}",  last_reported_time);
            }
            self.training_loader.try_switch();
            sleep(Duration::from_millis(2000));
        }
        debug!("booster, first sample is loaded");
        last_reported_time = timer.get_duration();
        while self.base_model_sig == MODEL_SIG_PLACEHOLDER {
            if timer.get_duration() - last_reported_time > 60.0 {
                last_reported_time = timer.get_duration();
                debug!("scanner, init, waiting the initial model, {}", last_reported_time);
            }
            self.handle_network(false);
            sleep(Duration::from_secs(2));
        }
        debug!("booster, remote initial model is downloaded");
    }

    fn set_root_tree(&mut self) {
        let max_sample_size = self.max_sample_size;
        let (_, base_pred, base_gamma) = get_base_node(max_sample_size, &mut self.training_loader);
        self.model.add_root(base_pred, base_gamma);
        info!("scanner, added new rule, {}, {}, {}, {}, {}",
              self.model.size(), max_sample_size, max_sample_size, 0, 0);
    }

    fn update_model(&mut self, model: Model, model_sig: String, source: &str) {
        let (old_size, old_base_size) = (self.model.size(), self.base_model_size);
        self.model = model;
        self.base_model_sig = model_sig;
        self.base_model_size = self.model.size();
        self.last_sent_model_length = self.model.size();
        if self.model.tree_size == 0 {
            self.set_root_tree();
        }
        if old_size > old_base_size {
            // loader needs to get rid of the rules that just got overwritten
            self.training_loader.reset_scores();
        }
        self.learner.reset();
        debug!("model-replaced, {}, {}, {}, {}",
                self.model.size(), old_size, self.base_model_sig, source);
    }

    /// Enable network communication. `name` is the name of this worker, which can be arbitrary
    /// and is only used for debugging purpose.
    /// `port` is the port number that used for network communication.
    pub fn enable_network(&mut self, name: String, port: u16) {
        let (local_s, local_r): (mpsc::Sender<Packet>, mpsc::Receiver<Packet>) =
            mpsc::channel();
        start_network_only_send(name.as_ref(), port, local_r).unwrap();
        // let (hb_s, hb_r): (mpsc::Sender<String>, mpsc::Receiver<String>) = mpsc::channel();
        // start_network_only_send(name.as_ref(), port + 1, hb_r);
        self.network_sender = Some(local_s);
        self.local_name = name.clone();
        self.local_id = {
            let t: Vec<&str> = self.local_name.rsplitn(2, '_').collect();
            t[0].parse().unwrap()
        };
    }

    /// Start training the boosting algorithm.
    pub fn training(
        &mut self,
        prep_time: f32,
    ) {
        debug!("Start training.");
        self.init();
        debug!("Finished initialization");

        let mut global_timer = PerformanceMonitor::new();
        let mut learner_timer = PerformanceMonitor::new();
        global_timer.start();
        let mut _last_communication_ts = global_timer.get_duration();
        let mut last_logging_ts = global_timer.get_duration();

        let mut total_data_size_without_fire = 0;
        self.verbose = false;
        while self.learner.is_gamma_significant() &&
                (self.num_trees <= 0 || self.model.tree_size < self.num_trees) {
            // Logging for the status check
            if global_timer.get_duration() - last_logging_ts >= 10.0 {
                self.print_log(total_data_size_without_fire);
                last_logging_ts = global_timer.get_duration();
            }

            let (new_rule, batch_size, switched) = {
                let (data, switched) =
                    self.training_loader.get_next_batch_and_update(true, &self.model);
                if switched {
                    // TODO: it seems we don't we have to reset the learner for the new sample
                    self.learner.reset();
                }
                learner_timer.resume();
                let new_rule = self.learner.update(&self.model, &data);
                learner_timer.update(data.len());
                learner_timer.pause();
                (new_rule, data.len(), switched)
            };
            total_data_size_without_fire += batch_size;
            global_timer.update(batch_size);

            // if exhausted all examples, return the emprically best one
            let is_full_scan = total_data_size_without_fire >= self.training_loader.size;
            let new_rule = {
                if new_rule.is_none() && is_full_scan {
                    Some(self.learner.get_max_empirical_ratio_tree_node())
                } else {
                    new_rule
                }
            };

            // Try to find new rule
            if switched {
                self.is_sample_version_changed = true;
                self.update_model(
                    self.training_loader.base_model.clone(),
                    self.training_loader.base_model_sig.clone(),
                    "loader",
                );
            }
            if new_rule.is_some() {
                let ts = prep_time + global_timer.get_duration();
                self.process_new_rule(new_rule.unwrap(), total_data_size_without_fire, ts);
                total_data_size_without_fire = 0;
            }

            // Communicate
            let is_communicated = self.handle_network(
                total_data_size_without_fire >= self.training_loader.size);
            if is_communicated {
                total_data_size_without_fire = 0;
                _last_communication_ts = global_timer.get_duration();
            }

            global_timer.write_log("boosting-overall");
            learner_timer.write_log("boosting-learning");
        }
        self.handle_persistent(prep_time + global_timer.get_duration());
        info!("Training is finished. Model length: {}. Is gamma significant? {}.",
              self.model.size(), self.learner.is_gamma_significant());
    }

    fn process_new_rule(
        &mut self, rule: TreeNode, total_data_size: usize, ts: f32,
    ) {
        rule.write_log();
        let (left_index, right_index) = self.model.add_nodes(
            rule.prt_index,
            rule.feature,
            rule.threshold,
            rule.predict,
            rule.gamma,
        );
        info!("scanner, added new rule, {}, {}, {}, {}, {}",
                self.model.size(), rule.num_scanned, total_data_size, left_index, right_index);
        // post updates
        self.learner.reset();
        if self.model.size() % self.save_interval == 0 {
            self.handle_persistent(ts);
        }
        self.is_scanner_status_changed = true;
    }

    // return true if a packet is sent out
    fn handle_network(&mut self, full_scanned: bool) -> bool {
        if self.network_sender.is_none() {
            return false;
        }
        let pack = download_model(&self.exp_name);
        let ret = {
            if pack.is_some() {
                let (r_model, r_model_sig, current_gamma): ModelPack = pack.unwrap();
                let is_packet_sent = self.check_and_send_packet(r_model, r_model_sig, full_scanned);
                if self.learner.set_gamma(current_gamma) {
                    self.is_scanner_status_changed = true;
                }
                is_packet_sent
            } else {
                debug!("booster, download-model, failed");
                false
            }
        };
        self.update_assignment();
        ret
    }

    // return true if a new packet is sent out
    fn check_and_send_packet(
        &mut self, remote_model: Model, remote_model_sig: String, full_scanned_no_update: bool,
    ) -> bool {
        // If it is newer, overwrite local model
        // Otherwise, push the current update to remote
        if remote_model_sig != self.base_model_sig {
            self.update_model(remote_model, remote_model_sig, "network");
            false
        } else if self.model.size() > self.last_sent_model_length ||
                    self.is_sample_version_changed {
            // send out the local patch
            self.send_packet();
            self.is_sample_version_changed = false;
            debug!("scanner, send-message, nonempty, {}, {}",
                    self.model.size() - self.base_model_size, self.model.size());
            true
        } else if self.model.size() == self.base_model_size &&
                    full_scanned_no_update && self.is_scanner_status_changed {
            // send out the empty message
            self.send_packet();
            self.is_scanner_status_changed = false;
            debug!("scanner, send-message, empty, {}, {}",
                    self.model.size() - self.last_sent_model_length, self.model.size());
            true
        } else {
            false
        }
    }

    fn send_packet(&mut self) -> bool {
        self.packet_counter += 1;
        let tree_slice = self.model.model_updates.create_slice(
            self.base_model_size..self.model.size());
        let gamma = self.learner.rho_gamma;
        let packet = Packet::new(
            &self.local_name,
            self.local_id,
            self.packet_counter,
            self.learner.expand_node,
            self.model.size(),
            tree_slice,
            gamma,
            self.training_loader.ess,
            self.training_loader.current_version,
            self.base_model_sig.clone(),
        );
        let send_result = self.network_sender.as_ref().unwrap()
                                .send(packet);
        if let Err(err) = send_result {
            error!("Attempt to send the packet to the network module but failed.
                    Error: {}", err);
            false
        } else {
            info!("Sent the local model to the network module");
            self.last_sent_model_length = self.model.size();
            true
        }
    }

    fn update_assignment(&mut self) {
        let assigns = download_assignments(&self.exp_name);
        if assigns.is_some() {
            let assignments = assigns.unwrap();
            let expand_node = assignments[self.local_id % assignments.len()];
            self.is_assignment_none = expand_node.is_none();
            if expand_node.is_some() {
                if self.learner.set_expand_node(expand_node.unwrap()) {
                    self.is_scanner_status_changed = true;
                    self.learner.reset();
                }
            }
        }
    }

    fn handle_persistent(&mut self, timestamp: f32) {
        self.persist_id += 1;
        write_model(&self.model, timestamp, self.save_process);
    }

    fn print_log(&self, total_data_size_without_fire: usize) {
        debug!("booster, status, {}",
                vec![
                    self.base_model_sig.to_string(),
                    self.model.size().to_string(),
                    self.last_sent_model_length.to_string(),
                    total_data_size_without_fire.to_string(),
                    self.learner.rho_gamma.to_string(),
                    self.is_scanner_status_changed.to_string(),
                    self.is_sample_version_changed.to_string(),
                ].join(", ")
        );
    }

    #[allow(dead_code)]
    fn print_verbose_log<T>(&self, message: T) where T: Display {
        if self.verbose {
            debug!("booster, verbose, {}", message);
        }
    }
}