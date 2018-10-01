mod bins;
mod learner;

extern crate serde_json;

use rayon::prelude::*;

use std::ops::Range;
use std::sync::mpsc;
use std::sync::mpsc::Sender;
use std::sync::mpsc::Receiver;

use tmsn::network::start_network;

use buffer_loader::BufferLoader;
use tree::Tree;
use commons::Model;
use commons::performance_monitor::PerformanceMonitor;
use commons::ModelScore;
use self::learner::Learner;

use self::bins::create_bins;
use commons::get_relative_weights;
use commons::get_symmetric_label;
use commons::is_positive;
use commons::io::create_bufwriter;
use commons::io::write_to_text_file;


/// The boosting algorithm. It contains two functions, one for starting
/// the network communication, the other for starting the training procedure.
pub struct Boosting {
    training_loader: BufferLoader,

    learner: Learner,
    model: Model,
    last_backup_time: f32,
    persist_id: u32,

    network_sender: Option<Sender<ModelScore>>,
    network_receiver: Option<Receiver<ModelScore>>,
    sum_gamma: f32,
    prev_sum_gamma: f32,

    sampler_send: Sender<Model>,
    validator_send: Sender<Model>,
}

impl Boosting {
    /// Create a boosting training class.
    ///
    /// * `training_loader`: the double-buffered data loader that provides examples to the algorithm.
    /// * `range`: the range of the feature dimensions that the weak rules would be selected from. In most cases,
    /// if the RustBoost is running on a single worker, `range` is equal to the `0..feature_size`; if it is running
    /// over multiple workers, it might be a subset of the full feature set.
    /// * `max_sample_size`: the number of examples to scan for determining the percentiles for the features.
    /// * `max_bin_size`: the size of the percentiles to generate on each feature dimension.
    /// * `default_gamma`: the initial value of the edge `gamma` of the candidate valid weak rules.
    pub fn new(
                mut training_loader: BufferLoader,
                range: Range<usize>,
                max_sample_size: usize,
                max_bin_size: usize,
                default_gamma: f32,
                sampler_send: Sender<Model>,
                validator_send: Sender<Model>,
            ) -> Boosting {
        let bins = create_bins(max_sample_size, max_bin_size, &range, &mut training_loader);
        let learner = Learner::new(default_gamma, bins, &range);

        // add root node for balancing labels
        let (base_tree, gamma) = get_base_tree(max_sample_size, &mut training_loader);
        let gamma_squared = gamma.powi(2);
        let model = vec![base_tree];

        Boosting {
            training_loader: training_loader,

            learner: learner,
            model: model,
            last_backup_time: 0.0,
            persist_id: 0,

            network_sender: None,
            network_receiver: None,
            sum_gamma: gamma_squared.clone(),
            prev_sum_gamma: gamma_squared,

            sampler_send: sampler_send,
            validator_send: validator_send,
        }
    }


    /// Enable network communication. `name` is the name of this worker, which can be arbitrary
    /// and is only used for debugging purpose. `remote_ips` is the vector of IPs of neighbor workers.
    /// `port` is the port number that used for network communication.
    pub fn enable_network(&mut self, name: String, remote_ips: &Vec<String>, port: u16) {
        let (local_send, local_recv): (Sender<ModelScore>, Receiver<ModelScore>) = mpsc::channel();
        let (other_send, other_recv): (Sender<ModelScore>, Receiver<ModelScore>) = mpsc::channel();
        self.network_sender = Some(local_send);
        self.network_receiver = Some(other_recv);
        start_network(name.as_ref(), remote_ips, port, true, other_send, local_recv);
    }


    /// Start training the boosting algorithm.
    ///
    /// * `num_iterations`: the number of boosting iteration. If it equals to 0, then the algorithm runs indefinitely.
    /// * `max_trials_before_shrink`: if cannot find any valid weak rules after scanning `max_trials_before_shrink` number of
    /// examples, shrinking the value of the targetting edge `gamma` of the weak rule.
    pub fn training(
            &mut self,
            num_iterations: usize,
            max_trials_before_shrink: u32) {
        info!("Start training.");
        let timeout = max_trials_before_shrink as usize;
        let mut global_timer = PerformanceMonitor::new();
        let mut learner_timer = PerformanceMonitor::new();
        global_timer.start();

        let mut model_ts = global_timer.get_duration();

        let mut scanned_counter = 0;
        while num_iterations <= 0 || self.model.len() < num_iterations {
            if self.learner.get_count() >= timeout {
                self.learner.shrink_target();
            }
            // if self.learner.get_gamma() <= 0.0001 {
            //     break;
            // }

            {
                self.training_loader.fetch_next_batch(true);
                self.training_loader.update_scores(&self.model);
                let data = self.training_loader.get_curr_batch(true);
                let weights = get_relative_weights(data);
                learner_timer.resume();
                self.learner.update(data, &weights);

                let scanned = data.len();
                scanned_counter += scanned;
                learner_timer.update(scanned);
                global_timer.update(scanned);
                learner_timer.pause();
            }

            let found_new_rule =
                if let &Some(ref weak_rule) = self.learner.get_new_weak_rule() {
                    self.model.push(
                        weak_rule.create_tree()
                    );
                    true
                } else {
                    false
                };
            if found_new_rule {
                self.sum_gamma += self.learner.get_gamma().powi(2);
                self.try_send_model();
                info!(
                    "new-tree-info, {}, {}, {}, {}, {}",
                    self.model.len(),
                    self.learner.get_count(),
                    self.learner.get_gamma(),
                    self.sum_gamma,
                    scanned_counter
                );
                scanned_counter = 0;
                model_ts = global_timer.get_duration();
                self.learner.reset();
            }

            if self.handle_network() {
                model_ts = global_timer.get_duration();
            }
            if model_ts - self.last_backup_time >= 5.0 {
                self.handle_persistent(model_ts);
                self.last_backup_time = model_ts;
            }

            if global_timer.write_log("boosting-overall") {
                learner_timer.write_log("boosting-learning");
            }
        }
        info!("Model in JSON:\n{}", serde_json::to_string(&self.model).unwrap());
        if model_ts - self.last_backup_time > 1e-8 {
            self.handle_persistent(model_ts);
            self.last_backup_time = model_ts;
        }
    }

    fn handle_network(&mut self) -> bool {
        let mut replaced = false;
        if self.network_receiver.is_some() {
            // info!("Processing models received from the network");
            // handle receiving
            let recv = self.network_receiver.as_ref().unwrap();  // safe, guaranteed in the IF statement
            let mut best_model = None;
            let mut max_score = 0.0;
            // process all models received so far
            while let Some(model_score) = recv.try_iter().next() {
                let (new_model, sum_gamma) = model_score;
                if best_model.is_none() || sum_gamma > max_score {
                    best_model = Some(new_model);
                    max_score = sum_gamma;
                }
            }
            if max_score > self.sum_gamma {
                let old_model_size = self.model.len();
                let old_model_score = self.sum_gamma;
                self.model = best_model.unwrap();  // safe
                self.sum_gamma = max_score;
                self.prev_sum_gamma = self.sum_gamma;
                self.learner.reset();
                replaced = true;
                debug!("model-replaced, {}, {}, {}, {}",
                       self.sum_gamma, old_model_score, self.model.len(), old_model_size);
            } else {
                // info!("Remote models are not better. Skipped.");
            }

            // handle sending
            if self.sum_gamma > self.prev_sum_gamma {
                let send_result = self.network_sender.as_ref().unwrap()
                                      .send((self.model.clone(), self.sum_gamma));
                if let Err(err) = send_result {
                    error!("Attempt to send the local model
                            to the network module but failed. Error: {}", err);
                } else {
                    info!("Sent the local model to the network module, {}, {}",
                          self.prev_sum_gamma, self.sum_gamma);
                    self.prev_sum_gamma = self.sum_gamma;
                }
            }
        }
        replaced
    }

    fn handle_persistent(&mut self, ts: f32) {
        let json = serde_json::to_string(&(ts, &self.model)).expect(
            "Local model cannot be serialized."
        );
        let mut file_buffer = create_bufwriter(&format!("model-{}.json", self.persist_id));
        self.persist_id += 1;
        write_to_text_file(&mut file_buffer, &json);
        // info!("Model {} is write to disk.", self.last_backup);
    }

    fn try_send_model(&mut self) {
        if let Some(ref mut network_sender) = self.network_sender {
            network_sender.send((self.model.clone(), self.sum_gamma)).unwrap();
        }
        self.sampler_send.send(self.model.clone()).unwrap();
        self.validator_send.send(self.model.clone()).unwrap();
    }
}


fn get_base_tree(max_sample_size: usize, data_loader: &mut BufferLoader) -> (Tree, f32) {
    let mut remaining_reads = max_sample_size;
    let mut n_pos = 0;
    let mut n_neg = 0;
    while remaining_reads > 0 {
        data_loader.fetch_next_batch(true);
        let data = data_loader.get_curr_batch(false /* skip scores update */);
        let (num_pos, num_neg) = data.par_iter().map(|example| {
            if is_positive(&get_symmetric_label(&example.0)) {
                (1, 0)
            } else {
                (0, 1)
            }
        }).reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        n_pos += num_pos;
        n_neg += num_neg;
        remaining_reads -= num_pos + num_neg;
    }

    let gamma = (0.5 - n_pos as f32 / (n_pos + n_neg) as f32).abs();
    let prediction = 0.5 * (n_pos as f32 / n_neg as f32).ln();
    let mut tree = Tree::new(2);
    tree.split(0, 0, 0.0, prediction, prediction);
    tree.release();

    info!("Root tree is added.");
    debug!("new-tree-info, {}, {}, {}, {}, \"{:?}\"", 1, max_sample_size, gamma, gamma * gamma, tree);
    (tree, gamma)
}
