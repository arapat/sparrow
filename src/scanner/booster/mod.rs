pub mod learner;
pub mod learner_helpers;

use std::sync::Arc;
use std::sync::RwLock;
use std::fmt::Display;

use commons::persistent_io::write_model;
use self::learner_helpers::get_base_node;

use config::Config;
use commons::Model;
use scanner::buffer_loader::BufferLoader;
use commons::bins::Bins;
use commons::packet::TaskPacket;
use commons::performance_monitor::PerformanceMonitor;
use self::learner::Learner;
use super::BoosterState;



/// The boosting algorithm. It contains two functions, one for starting
/// the network communication, the other for starting the training procedure.
pub struct Boosting {
    booster_state: Arc<RwLock<BoosterState>>,
    training_loader: BufferLoader,
    learner: Learner,

    init_packet: TaskPacket,
    model: Model,
    expand_node: usize,
    gamma: f32,

    max_sample_size: usize,
    save_process: bool,
    verbose: bool,
}

impl Boosting {
    /// Create a boosting training class.
    ///
    /// * `training_loader`: the double-buffered data loader that provides examples to the algorithm.
    /// over multiple workers, it might be a subset of the full feature set.
    /// * `max_sample_size`: the number of examples to scan for determining the percentiles for the features.
    /// * `default_gamma`: the initial value of the edge `gamma` of the candidate valid weak rules.
    pub fn new(
        init_packet: TaskPacket,
        booster_state: Arc<RwLock<BoosterState>>,
        training_loader: BufferLoader,
        bins: Vec<Bins>,
        config: &Config,
    ) -> Boosting {
        // TODO: make num_cadid a paramter
        let packet = init_packet.clone();
        let (model, gamma, expand_node) = (
            packet.model.unwrap(), packet.gamma.unwrap(), packet.expand_node.unwrap(),
        );
        let mut learner = Learner::new(gamma, bins, config.num_features, config.num_splits);
        learner.set_expand_node(expand_node);
        Boosting {
            booster_state: booster_state,
            training_loader: training_loader,
            learner: learner,

            init_packet: init_packet,
            model: model,
            expand_node: expand_node,
            gamma: gamma,

            max_sample_size: config.max_sample_size,
            save_process: config.save_process,
            verbose: false,
        }
    }

    pub fn destroy(self) -> (TaskPacket, Model, BufferLoader) {
        (self.init_packet, self.model, self.training_loader)
    }

    /// Start training the boosting algorithm.
    pub fn training(&mut self) {
        if self.model.tree_size == 0 {
            self.set_root_tree();
            return;
        }
        debug!("Start training.");

        let mut global_timer = PerformanceMonitor::new();
        let mut learner_timer = PerformanceMonitor::new();
        global_timer.start();
        let mut last_logging_ts = global_timer.get_duration();

        let mut booster_state: BoosterState = BoosterState::RUNNING;
        let mut data_scanned = 0;
        let mut new_rule = None;
        self.verbose = false;
        while booster_state == BoosterState::RUNNING && data_scanned < self.training_loader.size {
            // Logging for the status check
            if global_timer.get_duration() - last_logging_ts >= 10.0 {
                self.print_log(data_scanned);
                last_logging_ts = global_timer.get_duration();
            }

            let (rule, batch_size, switched) = {
                let (data, switched) =
                    self.training_loader.get_next_batch_and_update(true, &self.model);

                learner_timer.resume();
                let new_rule = self.learner.update(&self.model, &data);
                learner_timer.update(data.len());
                learner_timer.pause();

                (new_rule, data.len(), switched)
            };
            data_scanned += batch_size;
            global_timer.update(batch_size);

            // Try to find new rule
            if rule.is_some() {
                new_rule = rule;
                break;
            }

            global_timer.write_log("boosting-overall");
            learner_timer.write_log("boosting-learning");

            let state = self.booster_state.read().unwrap();
            booster_state = (*state).clone();
            drop(state);
        }
        let rule = new_rule.unwrap_or(self.learner.get_max_empirical_ratio_tree_node());
        rule.write_log();
        let (left_index, right_index) = self.model.add_nodes(
            rule.prt_index,
            rule.feature,
            rule.threshold,
            rule.predict,
            rule.gamma,
        );
        info!("scanner, added new rule, {}, {}, {}, {}, {}",
                self.model.size(), rule.num_scanned, data_scanned, left_index, right_index);
        write_model(&self.model, global_timer.get_duration(), self.save_process);
        info!("Training is finished. Model length: {}.", self.model.size());
    }

    fn set_root_tree(&mut self) {
        let max_sample_size = self.max_sample_size;
        let (_, base_pred, base_gamma) = get_base_node(max_sample_size, &mut self.training_loader);
        self.model.add_root(base_pred, base_gamma);
        info!("scanner, added new rule, {}, {}, {}, {}, {}",
              self.model.size(), max_sample_size, max_sample_size, 0, 0);
    }

    fn print_log(&self, data_scanned: usize) {
        debug!("booster, status, {}",
                vec![
                    self.model.size().to_string(),
                    data_scanned.to_string(),
                    self.learner.rho_gamma.to_string(),
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