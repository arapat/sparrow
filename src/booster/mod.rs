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

use commons::io::create_bufwriter;
use buffer_loader::BufferLoader;
use commons::Model;
use commons::performance_monitor::PerformanceMonitor;
use commons::ModelScore;
use commons::bins::Bins;
use commons::channel::Sender;
use tree::Tree;
use self::learner::get_base_node;
use self::learner::Learner;
use super::Example;


/// The boosting algorithm. It contains two functions, one for starting
/// the network communication, the other for starting the training procedure.
pub struct Boosting {
    num_iterations: usize,
    training_loader: BufferLoader,

    learner: Learner,
    model: Model,

    network_sender: Option<mpsc::Sender<ModelScore>>,
    network_receiver: Option<mpsc::Receiver<ModelScore>>,
    sum_gamma: f32,
    remote_sum_gamma: f32,

    sampler_channel_s: Sender<Model>,
    persist_id: u32,
    persist_file_buffer: Option<BufWriter<File>>,
    save_interval: usize,

    save_process: bool,
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
    /// * `default_gamma`: the initial value of the edge `gamma` of the candidate valid weak rules.
    pub fn new(
        num_iterations: usize,
        min_gamma: f32,
        max_trials_before_shrink: u32,
        training_loader: BufferLoader,
        // serial_training_loader: SerialStorage,
        bins: Vec<Bins>,
        range: Range<usize>,
        max_sample_size: usize,
        default_gamma: f32,
        sampler_channel_s: Sender<Model>,
        save_process: bool,
        save_interval: usize,
    ) -> Boosting {
        let mut training_loader = training_loader;
        let learner = Learner::new(
            min_gamma, default_gamma, max_trials_before_shrink, 10, bins, &range); // TODO: make num_cadid a paramter

        // add root node for balancing labels
        let (gamma, base_pred) = get_base_node(max_sample_size, &mut training_loader);
        let gamma_squared = gamma.powi(2);
        let model = Tree::new(num_iterations + 1, base_pred);

        let persist_file_buffer = {
            if save_process {
                None
            } else {
                Some(create_bufwriter(&String::from("model.json")))
            }
        };
        let mut b = Boosting {
            num_iterations: num_iterations,
            training_loader: training_loader,

            learner: learner,
            model: model,

            network_sender: None,
            network_receiver: None,
            sum_gamma: gamma_squared.clone(),
            remote_sum_gamma: gamma_squared,

            sampler_channel_s: sampler_channel_s,
            persist_id: 0,
            persist_file_buffer: persist_file_buffer,
            save_interval: save_interval,

            save_process: save_process,
        };
        b.try_send_model();
        b
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
    pub fn training(
        &mut self,
        prep_time: f32,
        _validate_set1: Vec<Example>,
        _validate_set2: Vec<Example>,
    ) {
        info!("Start training.");

        let init_sampling_duration = self.training_loader.get_sampling_duration();
        let mut global_timer = PerformanceMonitor::new();
        let mut learner_timer = PerformanceMonitor::new();
        global_timer.start();

        /*
        let mut validate_w1: Vec<f32> = {
            validate_set1.iter()
                        .map(|example| {
                            get_weight(example, self.model[0].get_leaf_prediction(example))
                        }).collect()
        };
        let mut validate_w2: Vec<f32> = {
            validate_set2.iter()
                        .map(|example| {
                            get_weight(example, self.model[0].get_leaf_prediction(example))
                        }).collect()
        };
        */
        let mut iteration = 0;
        let mut is_gamma_significant = true;
        while is_gamma_significant &&
                (self.num_iterations <= 0 || self.model.size < self.num_iterations) {
            let (new_rule, batch_size) = {
                let data = self.training_loader.get_next_batch_and_update(true, &self.model);
                learner_timer.resume();
                (self.learner.update(&self.model, &data), data.len())
            };
            learner_timer.update(batch_size);
            global_timer.update(batch_size);
            learner_timer.pause();

            if new_rule.is_some() {
                let new_rule = new_rule.unwrap();
                /*
                if validate_set1.len() > 0 {
                    validate_w1.par_iter_mut()
                              .zip(validate_set1.par_iter())
                              .for_each(|(w, example)| {
                                  *w *= get_weight(example, new_rule.get_leaf_prediction(example));
                              });
                    validate_w2.par_iter_mut()
                              .zip(validate_set2.par_iter())
                              .for_each(|(w, example)| {
                                  *w *= get_weight(example, new_rule.get_leaf_prediction(example));
                              });
                }
                */
                new_rule.write_log();
                let index = self.model.add_node(
                    new_rule.prt_index,
                    new_rule.feature,
                    new_rule.threshold,
                    new_rule.evaluation,
                    new_rule.predict,
                );
                let deactive = self.learner.push_active(index);
                if deactive.is_some() {
                    self.model.unmark_active(deactive.unwrap());
                }
                self.model.mark_active(index);

                // TODO: how to calculate sum_gamma in the case of trees?
                // self.sum_gamma += new_rule.gamma.powi(2);
                // post updates
                self.try_send_model();
                is_gamma_significant = self.learner.is_gamma_significant();
                self.learner.reset();
                if self.model.size % self.save_interval == 0 {
                    self.handle_persistent(iteration, prep_time + global_timer.get_duration());
                }

                info!("new-tree-info, {}", self.model.size);
            }

            iteration += 1;
            self.handle_network();

            let sampling_duration = self.training_loader.get_sampling_duration() - init_sampling_duration;
            global_timer.set_adjust(-sampling_duration);
            global_timer.write_log("boosting-overall");
            learner_timer.write_log("boosting-learning");
        }
        self.handle_persistent(iteration, prep_time + global_timer.get_duration());
        info!("Training is finished. Model length: {}. Is gamma significant? {}.",
              self.model.size, self.learner.is_gamma_significant());
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
            let (old_size, old_score) = (self.model.size, self.sum_gamma);
            self.model = model.unwrap();
            self.sum_gamma = score;
            self.remote_sum_gamma = self.sum_gamma;
            self.learner.reset();
            debug!("model-replaced, {}, {}, {}, {}",
                    self.sum_gamma, old_score, self.model.size, old_size);
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
                &format!("models/model_{}-v{}.json", self.model.size, self.persist_id));
            file_buffer.write(json.as_ref()).unwrap();
        } else {
            let buf = self.persist_file_buffer.as_mut().unwrap();
            buf.seek(SeekFrom::Start(0)).unwrap();
            buf.write(json.as_ref()).unwrap();
        }
    }

    fn try_send_model(&mut self) {
        // TODO: Activate network later
        // if let Some(ref mut network_sender) = self.network_sender {
        //     network_sender.send((self.model.clone(), self.sum_gamma)).unwrap();
        // }
        self.sampler_channel_s.try_send(self.model.clone());
    }
}
