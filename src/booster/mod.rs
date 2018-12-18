mod bins;
mod learner;

use std::fs::File;
use std::io::BufWriter;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::sync::mpsc;
use std::ops::Range;
use serde_json;
use tmsn::network::start_network;

use self::bins::create_bins;
use commons::io::create_bufwriter;
use buffer_loader::BufferLoader;
use stratified_storage::serial_storage::SerialStorage;
use commons::Model;
use commons::performance_monitor::PerformanceMonitor;
use commons::ModelScore;
use commons::channel::Sender;
use self::learner::get_base_tree;
use self::learner::Learner;


/// The boosting algorithm. It contains two functions, one for starting
/// the network communication, the other for starting the training procedure.
pub struct Boosting {
    num_iterations: usize,
    training_loader: BufferLoader,
    serial_training_loader: SerialStorage,
    // max_trials_before_shrink: u32,

    learner: Learner,
    model: Model,

    network_sender: Option<mpsc::Sender<ModelScore>>,
    network_receiver: Option<mpsc::Receiver<ModelScore>>,
    sum_gamma: f32,
    remote_sum_gamma: f32,

    sampler_channel_s: Sender<Model>,
    persist_id: u32,
    persist_file_buffer: Option<BufWriter<File>>,

    save_process: bool,
    debug_mode: bool,
}

impl Boosting {
    /// Create a boosting training class.
    ///
    /// * `num_iterations`: the number of boosting iteration. If it equals to 0, then the algorithm runs indefinitely.
    /// * `max_trials_before_shrink`: if cannot find any valid weak rules after scanning `max_trials_before_shrink` number of
    /// examples, shrinking the value of the targetting edge `gamma` of the weak rule.
    /// * `training_loader`: the double-buffered data loader that provides examples to the algorithm.
    /// * `range`: the range of the feature dimensions that the weak rules would be selected from. In most cases,
    /// if the RustBoost is running on a single worker, `range` is equal to the `0..feature_size`; if it is running
    /// over multiple workers, it might be a subset of the full feature set.
    /// * `max_sample_size`: the number of examples to scan for determining the percentiles for the features.
    /// * `max_bin_size`: the size of the percentiles to generate on each feature dimension.
    /// * `default_gamma`: the initial value of the edge `gamma` of the candidate valid weak rules.
    pub fn new(
        num_iterations: usize,
        max_leaves: usize,
        min_gamma: f32,
        max_trials_before_shrink: u32,
        training_loader: BufferLoader,
        serial_training_loader: SerialStorage,
        range: Range<usize>,
        max_sample_size: usize,
        max_bin_size: usize,
        default_gamma: f32,
        sampler_channel_s: Sender<Model>,
        save_process: bool,
        debug_mode: bool,
    ) -> Boosting {
        let mut training_loader = training_loader;
        let bins = create_bins(max_sample_size, max_bin_size, &range, &mut training_loader);
        let learner = Learner::new(
            max_leaves, min_gamma, default_gamma, max_trials_before_shrink, bins, &range);

        // add root node for balancing labels
        let (base_tree, gamma) = get_base_tree(max_sample_size, &mut training_loader);
        let gamma_squared = gamma.powi(2);
        let model = vec![base_tree];

        let persist_file_buffer = {
            if save_process {
                None
            } else {
                Some(create_bufwriter(&String::from("model.json")))
            }
        };
        Boosting {
            num_iterations: num_iterations,
            training_loader: training_loader,
            serial_training_loader: serial_training_loader,
            // max_trials_before_shrink: max_trials_before_shrink,

            learner: learner,
            model: model,

            network_sender: None,
            network_receiver: None,
            sum_gamma: gamma_squared.clone(),
            remote_sum_gamma: gamma_squared,

            sampler_channel_s: sampler_channel_s,
            persist_id: 0,
            persist_file_buffer: persist_file_buffer,

            save_process: save_process,
            debug_mode: debug_mode,
        }
    }


    /// Enable network communication. `name` is the name of this worker, which can be arbitrary
    /// and is only used for debugging purpose. `remote_ips` is the vector of IPs of neighbor workers.
    /// `port` is the port number that used for network communication.
    pub fn enable_network(
        &mut self,
        name: String,
        remote_ips: &Vec<String>,
        port: u16,
    ) {
        let (local_s, local_r): (mpsc::Sender<ModelScore>, mpsc::Receiver<ModelScore>) =
            mpsc::channel();
        let (remote_s, remote_r): (mpsc::Sender<ModelScore>, mpsc::Receiver<ModelScore>) =
            mpsc::channel();
        self.network_sender = Some(local_s);
        self.network_receiver = Some(remote_r);
        start_network(name.as_ref(), remote_ips, port, true, remote_s, local_r);
    }


    /// Start training the boosting algorithm.
    pub fn training(&mut self) {
        info!("Start training.");

        let init_sampling_duration = self.training_loader.get_sampling_duration();
        let mut global_timer = PerformanceMonitor::new();
        let mut learner_timer = PerformanceMonitor::new();
        global_timer.start();

        let mut iteration = 0;
        while self.learner.is_any_candidate_active() &&
                (self.num_iterations <= 0 || self.model.len() < self.num_iterations) {
            let (new_rule, batch_size) = {
                let data = self.training_loader.get_next_batch_and_update(true, &self.model);
                learner_timer.resume();
                (self.learner.update(data), data.len())
            };
            learner_timer.update(batch_size);
            global_timer.update(batch_size);
            learner_timer.pause();

            if new_rule.is_some() {
                let new_rule = new_rule.unwrap();
                self.model.push(new_rule);
                // TODO: how to calculate sum_gamma in the case of trees?
                // self.sum_gamma += new_rule.gamma.powi(2);
                // post updates
                self.try_send_model();
                self.learner.reset_all();
                info!("new-tree-info, {}", self.model.len());
                if self.model.len() % 10 == 0 {
                    self.handle_persistent(iteration, global_timer.get_duration());
                }

                if self.debug_mode {
                    // TODO: tidy up this debugging code; support general loss function
                    let mut k = 0;
                    let mut score: f32 = 0.0;
                    while k < self.serial_training_loader.size {
                        let examples = self.serial_training_loader.read(1000);
                        k += examples.len();
                        let sum_scores: f32 = examples.iter().map(|example| {
                            let pred: f32 = self.model.iter()
                                           .map(|model| model.get_leaf_prediction(example)).sum();
                            let label: f32 = example.label as f32;  // either +1 or -1
                            (-label * pred).exp()
                        }).sum();
                        score += sum_scores;
                    }
                    debug!("Validation: {}", score / (k as f32));
                }
            }

            iteration += 1;
            self.handle_network();

            let sampling_duration = self.training_loader.get_sampling_duration() - init_sampling_duration;
            global_timer.set_adjust(-sampling_duration);
            global_timer.write_log("boosting-overall");
            learner_timer.write_log("boosting-learning");
        }
        self.handle_persistent(iteration, global_timer.get_duration());
        info!("Training is finished.");
    }

    fn handle_network(&mut self) -> bool {
        if self.network_receiver.is_none() {
            return false;
        }
        // process all models received so far
        let (model, score) = self.network_receiver.as_ref().unwrap().try_iter().fold(
            (None, self.sum_gamma),
            |cur_best, model_score| {
                let (cur_model, cur_score) = cur_best;
                let (new_model, score) = model_score;
                if cur_model.is_none() || cur_score < score {
                    (Some(new_model), score)
                } else {
                    (cur_model, cur_score)
                }
            }
        );
        let replace = model.is_some();
        if replace {
            let (old_size, old_score) = (self.model.len(), self.sum_gamma);
            self.model = model.unwrap();
            self.sum_gamma = score;
            self.remote_sum_gamma = self.sum_gamma;
            self.learner.reset_all();
            debug!("model-replaced, {}, {}, {}, {}",
                    self.sum_gamma, old_score, self.model.len(), old_size);
        }

        // handle sending
        if self.sum_gamma > self.remote_sum_gamma {
            let send_result = self.network_sender.as_ref().unwrap()
                                    .send((self.model.clone(), self.sum_gamma));
            if let Err(err) = send_result {
                error!("Attempt to send the local model to the network module but failed.
                        Error: {}", err);
            } else {
                info!("Sent the local model to the network module, {}, {}",
                        self.remote_sum_gamma, self.sum_gamma);
                self.remote_sum_gamma = self.sum_gamma;
            }
        }
        replace
    }

    fn handle_persistent(&mut self, iteration: usize, timestamp: f32) {
        let json = serde_json::to_string(&(timestamp, iteration, &self.model)).expect(
            "Local model cannot be serialized."
        );
        self.persist_id += 1;
        if self.save_process {
            let mut file_buffer = create_bufwriter(
                &format!("models/model_{}-v{}.json", self.model.len(), self.persist_id));
            file_buffer.write(json.as_ref()).unwrap();
        } else {
            let buf = self.persist_file_buffer.as_mut().unwrap();
            buf.seek(SeekFrom::Start(0)).unwrap();
            buf.write(json.as_ref()).unwrap();
        }
    }

    fn try_send_model(&mut self) {
        if let Some(ref mut network_sender) = self.network_sender {
            network_sender.send((self.model.clone(), self.sum_gamma)).unwrap();
        }
        self.sampler_channel_s.try_send(self.model.clone());
    }
}
