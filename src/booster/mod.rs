mod learner;

use std::fs::File;
use std::io::BufWriter;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::sync::mpsc;
use std::ops::Range;
use serde_json;
use tmsn::network::start_network_only_send;

use buffer_loader::BufferLoader;
use commons::io::create_bufwriter;
use commons::Model;
use commons::performance_monitor::PerformanceMonitor;
use commons::ModelSig;
use commons::bins::Bins;
use commons::channel::Sender;
use model_sync::download_model;
use tree::Tree;
use tree::TreeSlice;
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
    model_sig: String,
    last_sent_model_sig: String,

    network_sender: Option<mpsc::Sender<ModelSig>>,
    local_name: String,
    last_remote_length: usize,

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
        let (_, base_pred, base_gamma) = get_base_node(max_sample_size, &mut training_loader);
        let model = Tree::new(num_iterations + 1, base_pred, base_gamma);

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
            model_sig: "".to_string(),
            last_sent_model_sig: ".".to_string(),

            network_sender: None,
            local_name: "".to_string(),
            last_remote_length: 0,

            sampler_channel_s: sampler_channel_s,
            persist_id: 0,
            persist_file_buffer: persist_file_buffer,
            save_interval: save_interval,

            save_process: save_process,
        };
        b.learner.push_active(0);
        b.model.mark_active(0);
        b.try_send_model();
        b
    }


    /// Enable network communication. `name` is the name of this worker, which can be arbitrary
    /// and is only used for debugging purpose.
    /// `port` is the port number that used for network communication.
    pub fn enable_network(&mut self, name: String, port: u16) {
        let (local_s, local_r): (mpsc::Sender<ModelSig>, mpsc::Receiver<ModelSig>) =
            mpsc::channel();
        start_network_only_send(name.as_ref(), port, local_r);
        self.network_sender = Some(local_s);
        self.local_name = name;
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
        let mut total_data_size = 0;
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
            total_data_size += batch_size;

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
                    new_rule.gamma,
                );
                info!("scanner, added new rule, {}, {}, {}",
                      self.model.size, new_rule.num_scanned, total_data_size);
                let deactive = self.learner.push_active(index);
                if deactive.is_some() {
                    self.model.unmark_active(deactive.unwrap());
                }
                self.model.mark_active(index);

                // post updates
                self.try_send_model();
                is_gamma_significant = self.learner.is_gamma_significant();
                self.learner.reset();
                if self.model.size % self.save_interval == 0 {
                    self.handle_persistent(iteration, prep_time + global_timer.get_duration());
                }
                total_data_size = 0;
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

    fn handle_network(&mut self) {
        if self.network_sender.is_none() {
            return;
        }

        // 0. Get the latest model
        // 1. If it is newer, overwrite local model
        // 2. Otherwise, push the current update to remote
        let model_score = download_model();
        if model_score.is_none() {
            return;
        }
        let (remote_model, model_sig): (Model, String) = model_score.unwrap();
        let new_model_sig = self.local_name.clone() + &self.model.size.to_string();
        if model_sig != self.model_sig {
            // replace the existing model
            let old_size = self.model.size;
            self.model = remote_model;
            self.model_sig = model_sig;
            self.last_remote_length = self.model.size;
            self.learner.reset();
            for i in 0..self.model.size {
                self.learner.push_active(i);
                self.model.mark_active(i);
            }
            debug!("model-replaced, {}, {}, {}", self.model.size, old_size, self.model_sig);
        } else if self.last_sent_model_sig != new_model_sig {
            // send out the local patch
            let tree_slice = TreeSlice::new(&self.model, self.last_remote_length..self.model.size);
            let packet: ModelSig =
                (tree_slice, self.model.last_gamma, model_sig, new_model_sig.clone());
            let send_result = self.network_sender.as_ref().unwrap()
                                    .send(packet);
            if let Err(err) = send_result {
                error!("Attempt to send the local model to the network module but failed.
                        Error: {}", err);
            } else {
                self.last_sent_model_sig = new_model_sig;
                info!("Sent the local model to the network module, {}, {}",
                      self.last_sent_model_sig, self.model.size);
            }
        }
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
