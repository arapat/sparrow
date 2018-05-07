extern crate serde_json;

use rayon::prelude::*;

use std::ops::Range;
use std::sync::mpsc;
use std::sync::mpsc::Sender;
use std::sync::mpsc::Receiver;

use data_loader::DataLoader;
use learner::Learner;
use tree::Tree;
use commons::Model;
use commons::LossFunc;
use commons::PerformanceMonitor;
use commons::ModelScore;

use bins::create_bins;
use commons::get_weights;
use commons::get_symmetric_label;
use commons::is_positive;
use data_loader::io::create_bufwriter;
use data_loader::io::write_to_text_file;
use network::start_network;
use validator::validate;


pub struct Boosting<'a> {
    training_loader_stack: Vec<DataLoader>,
    testing_loader: DataLoader,
    eval_funcs: Vec<&'a LossFunc>,

    sample_ratio: f32,
    ess_threshold: f32,

    learner: Learner,
    model: Model,
    last_backup: usize,

    sender: Option<Sender<ModelScore>>,
    receiver: Option<Receiver<ModelScore>>,
    sum_gamma: f32,
    prev_sum_gamma: f32
}

impl<'a> Boosting<'a> {
    pub fn new(
                mut training_loader: DataLoader,
                testing_loader: DataLoader,
                range: Range<usize>,
                max_sample_size: usize,
                max_bin_size: usize,
                sample_ratio: f32,
                ess_threshold: f32,
                default_rho_gamma: f32,
                eval_funcs: Vec<&'a LossFunc>
            ) -> Boosting<'a> {
        let bins = create_bins(max_sample_size, max_bin_size, &range, &mut training_loader);
        let learner = Learner::new(default_rho_gamma, bins, &range);

        // add root node for balancing labels
        let (base_tree, gamma) = get_base_tree(max_sample_size, &mut training_loader);
        let gamma_squared = gamma.powi(2);
        let model = vec![base_tree];

        Boosting {
            training_loader_stack: vec![training_loader],
            testing_loader: testing_loader,
            eval_funcs: eval_funcs,

            sample_ratio: sample_ratio,
            ess_threshold: ess_threshold,

            learner: learner,
            model: model,
            last_backup: 0,

            sender: None,
            receiver: None,
            sum_gamma: gamma_squared.clone(),
            prev_sum_gamma: gamma_squared
        }
    }


    pub fn enable_network(&mut self, name: String, remote_ips: &Vec<String>, port: u16) {
        let (local_send, local_recv): (Sender<ModelScore>, Receiver<ModelScore>) = mpsc::channel();
        let (other_send, other_recv): (Sender<ModelScore>, Receiver<ModelScore>) = mpsc::channel();
        self.sender = Some(local_send);
        self.receiver = Some(other_recv);
        start_network(name, remote_ips, port, other_send, local_recv)
    }

    pub fn training(
            &mut self,
            num_iterations: usize,
            max_trials_before_shrink: u32,
            validate_interval: u32) {
        info!("Start training.");
        let interval = validate_interval as usize;
        let timeout = max_trials_before_shrink as usize;
        let mut global_timer = PerformanceMonitor::new();
        let mut sampler_timer = PerformanceMonitor::new();
        let mut learner_timer = PerformanceMonitor::new();
        global_timer.start();

        let speed_test = false;
        let mut speed_read = 0;

        if self.training_loader_stack.len() <= 1 {
            info!("Initial sampling.");
            self.sample(&mut sampler_timer);
            global_timer.update(sampler_timer.get_performance().1);
        }

        while num_iterations <= 0 || self.model.len() < num_iterations {
            if !speed_test {
                let sampler_scanned = self.try_sample(&mut sampler_timer);
                global_timer.update(sampler_scanned);
            }

            let is_timeout = self.learner.get_count() >= timeout;
            if !is_timeout {
                let training_loader = &mut self.training_loader_stack[1];
                training_loader.fetch_next_batch();
                training_loader.fetch_scores(&self.model);
                let data = training_loader.get_curr_batch();
                let scores = training_loader.get_relative_scores();
                let weights = get_weights(data, scores);
                learner_timer.resume();
                self.learner.update(data, &weights);

                let scanned = data.len();
                learner_timer.update(scanned);
                global_timer.update(scanned);
                learner_timer.pause();
            }

            // if is_timeout {
            let mut new_tree = None;
            if let &Some(ref weak_rule) = self.learner.get_new_weak_rule(is_timeout) {
                new_tree = Some(weak_rule.create_tree());
            }
            if new_tree.is_some() {
                let this_tree = new_tree.unwrap();

                // Validation, only used in debugging
                /*
                let split_idx = this_tree.get_split_index();
                let split_val = this_tree.get_split_value();
                let num_batches = self.training_loader_stack[0].get_num_batches();
                let mut left_pos = 0.0;
                let mut left_neg = 0.0;
                let mut right_pos = 0.0;
                let mut right_neg = 0.0;
                (0..num_batches).for_each(|_| {
                    let all_data = &mut self.training_loader_stack[0];
                    all_data.fetch_next_batch();
                    all_data.fetch_scores(&self.model);
                    let data = all_data.get_curr_batch();
                    let labels: Vec<f32> = all_data.get_curr_batch()
                                .iter()
                                .map(|d| get_symmetric_label(d))
                                .collect();
                    let scores: Vec<f32> = all_data.get_absolute_scores()
                                .iter()
                                .cloned()
                                .collect();
                    let weights = get_weights(data, &scores);
                    data.iter().map(
                        |example| example.get_features()[split_idx]
                    ).zip(weights.iter()).zip(labels).for_each(|((feature_val, weight), label)| {
                        if feature_val as f32 <= split_val {
                            if label > 0.0 {
                                left_pos += weight;
                            } else {
                                left_neg += weight;
                            }
                        } else {
                            if label > 0.0 {
                                right_pos += weight;
                            } else {
                                right_neg += weight;
                            }
                        }
                    });
                });
                error!("verify, {}, {}, {}, {}", left_pos, left_neg, right_pos, right_neg);
                */
                // -- validation done --

                self.model.push(this_tree);
                self.sum_gamma += self.learner.get_rho_gamma().powi(2);
                debug!(
                    "new-tree-info, {}, {}, {}, {}, \"{:?}\"",
                    self.model.len(),
                    self.learner.get_count(),
                    self.learner.get_rho_gamma(),
                    self.sum_gamma,
                    self.model[self.model.len() - 1]
                );
                self.learner.reset();
                if interval > 0 && self.model.len() % interval == 0 {
                    self._validate();
                }
            }

            if is_timeout {
                self.learner.shrink_target();
            }

            self.handle_network();
            self.handle_persistent();
            // }

            let (since_last_check, count, duration, speed) = global_timer.get_performance();
            if speed_test || since_last_check >= 2 {
                let (_, count_learn, duration_learn, speed_learn) = learner_timer.get_performance();
                let (_, count_sampler, duration_sampler, speed_sampler) = sampler_timer.get_performance();
                debug!("boosting_speed, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                       self.model.len(), duration, count, speed,
                       duration_learn, count_learn, speed_learn,
                       duration_sampler, count_sampler, speed_sampler);
                global_timer.reset_last_check();

                speed_read += 1;
                if speed_test && speed_read >= 10 {
                    return;
                }
            }
        }
        info!("Model in JSON:\n{}", serde_json::to_string(&self.model).unwrap());
    }

    fn handle_network(&mut self) {
        if self.receiver.is_some() {
            info!("Processing models received from the network");
            // handle receiving
            let recv = self.receiver.as_ref().unwrap();  // safe, guaranteed in the IF statement
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
                debug!("model-replaced, {}, {}, {}, {}",
                       self.sum_gamma, old_model_score, self.model.len(), old_model_size);
            } else {
                info!("Remote models are not better. Skipped.");
            }

            // handle sending
            if self.sum_gamma > self.prev_sum_gamma {
                let send_result = self.sender.as_ref().unwrap()
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
    }

    fn handle_persistent(&mut self) {
        if self.model.len() - self.last_backup >= 100 {
            let json = serde_json::to_string(&self.model).expect(
                "Local model cannot be serialized."
            );
            let mut file_buffer = create_bufwriter(&format!("model-{}.json", self.model.len()));
            write_to_text_file(&mut file_buffer, &json);
            self.last_backup = self.model.len();
            info!("Model {} is write to disk.", self.last_backup);
        }
    }

    fn try_sample(&mut self, sampler_timer: &mut PerformanceMonitor) -> usize {
        let ess_option = self.training_loader_stack[1].get_ess();
        if let Some(ess) = ess_option {
            if ess < self.ess_threshold {
                debug!("resample for ESS too low, {}, {}", ess, self.ess_threshold);
                self.training_loader_stack.pop();
                let prev_count = sampler_timer.get_performance().1;
                self.sample(sampler_timer);
                self.learner.reset_all();
                return sampler_timer.get_performance().1 - prev_count;
            }
        }
        0
    }

    fn sample(&mut self, sampler_timer: &mut PerformanceMonitor) {
        info!("Re-sampling is started.");
        let new_sample = self.training_loader_stack[0].sample(&self.model, self.sample_ratio, sampler_timer);
        self.training_loader_stack.push(new_sample);
        info!("A new sample is generated.");
    }

    fn _validate(&mut self) {
        info!("Validation is started.");
        let scores = validate(&mut self.testing_loader, &self.model, &self.eval_funcs);
        let output: Vec<String> = scores.into_iter().map(|x| x.to_string()).collect();
        debug!("validation-results, {}, {}", self.model.len(), output.join(", "));
    }
}


fn get_base_tree(max_sample_size: usize, data_loader: &mut DataLoader) -> (Tree, f32) {
    let mut remaining_reads = max_sample_size;
    let mut n_pos = 0;
    let mut n_neg = 0;
    while remaining_reads > 0 {
        data_loader.fetch_next_batch();
        let data = data_loader.get_curr_batch();
        let (_p, _n) = data.par_iter().map(|example| {
            if is_positive(&get_symmetric_label(example)) {
                (1, 0)
            } else {
                (0, 1)
            }
        }).reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        n_pos += _p;
        n_neg += _n;
        remaining_reads -= _p + _n;
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
