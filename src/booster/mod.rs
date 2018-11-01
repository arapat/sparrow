mod bins;
mod learner;

extern crate serde_json;

use std::sync::mpsc;
use std::ops::Range;
use tmsn::network::start_network;

use self::bins::create_bins;
use commons::io::create_bufwriter;
use commons::io::write_to_text_file;
use buffer_loader::BufferLoader;
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
    // max_trials_before_shrink: u32,

    learner: Learner,
    model: Model,

    network_sender: Option<mpsc::Sender<ModelScore>>,
    network_receiver: Option<mpsc::Receiver<ModelScore>>,
    sum_gamma: f32,
    remote_sum_gamma: f32,

    sampler_channel_s: Sender<Model>,
    validator_channel_s: Sender<Model>,
    persist_id: u32,
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
        max_trials_before_shrink: u32,
        training_loader: BufferLoader,
        range: Range<usize>,
        max_sample_size: usize,
        max_bin_size: usize,
        default_gamma: f32,
        sampler_channel_s: Sender<Model>,
        validator_channel_s: Sender<Model>,
    ) -> Boosting {
        let mut training_loader = training_loader;
        let bins = create_bins(max_sample_size, max_bin_size, &range, &mut training_loader);
        let learner = Learner::new(default_gamma, max_trials_before_shrink, bins, &range);

        // add root node for balancing labels
        let (base_tree, gamma) = get_base_tree(max_sample_size, &mut training_loader);
        let gamma_squared = gamma.powi(2);
        let model = vec![base_tree];

        Boosting {
            num_iterations: num_iterations,
            training_loader: training_loader,
            // max_trials_before_shrink: max_trials_before_shrink,

            learner: learner,
            model: model,

            network_sender: None,
            network_receiver: None,
            sum_gamma: gamma_squared.clone(),
            remote_sum_gamma: gamma_squared,

            sampler_channel_s: sampler_channel_s,
            validator_channel_s: validator_channel_s,
            persist_id: 0,
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

        let mut global_timer = PerformanceMonitor::new();
        let mut learner_timer = PerformanceMonitor::new();
        global_timer.start();

        let mut iteration = 0;
        while self.num_iterations <= 0 || self.model.len() < self.num_iterations {
            learner_timer.resume();
            let (new_rule, batch_size) = {
                let data = self.training_loader.get_next_batch_and_update(true, &self.model);
                (self.learner.update(data), data.len())
            };
            learner_timer.update(batch_size);
            learner_timer.pause();

            if new_rule.is_some() {
                let new_rule = new_rule.unwrap();
                new_rule.write_log(self.model.len(), self.sum_gamma);
                self.sum_gamma += new_rule.gamma.powi(2);
                self.model.push(new_rule.to_tree());
                // post updates
                self.try_send_model();
                self.learner.reset();
            }
            iteration += 1;

            self.handle_network();
            if iteration % 10 == 0 {
                self.handle_persistent(iteration);
            }
            global_timer.update(batch_size);

            let sampling_duration = self.training_loader.get_sampling_duration();
            global_timer.set_adjust(-sampling_duration);
            global_timer.write_log("boosting-overall");
            learner_timer.set_adjust(-sampling_duration);
            learner_timer.write_log("boosting-learning");
        }
        self.handle_persistent(iteration);
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
            self.learner.reset();
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

    fn handle_persistent(&mut self, iteration: usize) {
        let json = serde_json::to_string(&(iteration, &self.model)).expect(
            "Local model cannot be serialized."
        );
        // let mut file_buffer = create_bufwriter(&format!("model-v{}.json", self.persist_id));
        let mut file_buffer = create_bufwriter(&String::from("model.json"));
        self.persist_id += 1;
        write_to_text_file(&mut file_buffer, &json);
    }

    fn try_send_model(&mut self) {
        if let Some(ref mut network_sender) = self.network_sender {
            network_sender.send((self.model.clone(), self.sum_gamma)).unwrap();
        }
        self.sampler_channel_s.try_send(self.model.clone());
        self.validator_channel_s.try_send(self.model.clone());
    }
}