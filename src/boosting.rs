extern crate serde_json;

use rayon::prelude::*;

use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc;
use std::sync::mpsc::Sender;
use std::sync::mpsc::Receiver;

use buffer_loader::BufferLoader;
use learner::Learner;
use tree::Tree;
use commons::Model;
use commons::LossFunc;
use commons::performance_monitor::PerformanceMonitor;
use commons::ModelScore;

use bins::create_bins;
use commons::get_relative_weights;
use commons::get_symmetric_label;
use commons::is_positive;
use commons::io::create_bufwriter;
use commons::io::write_to_text_file;
use network::start_network;


type NextLoader = Arc<Mutex<Option<BufferLoader>>>;
type ModelMutex = Arc<Mutex<Option<Model>>>;


pub struct Boosting<'a> {
    training_loader: BufferLoader,
    eval_funcs: Vec<&'a LossFunc>,

    sample_ratio: f32,
    ess_threshold: f32,

    learner: Learner,
    model: Model,
    last_backup_time: f32,
    persist_id: u32,

    // model_mutex: ModelMutex,

    sender: Option<Sender<ModelScore>>,
    receiver: Option<Receiver<ModelScore>>,
    sum_gamma: f32,
    prev_sum_gamma: f32
}

impl<'a> Boosting<'a> {
    pub fn new(
                mut training_loader: BufferLoader,
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
            training_loader: training_loader,
            eval_funcs: eval_funcs,

            sample_ratio: sample_ratio,
            ess_threshold: ess_threshold,

            learner: learner,
            model: model,
            last_backup_time: 0.0,
            persist_id: 0,

            // model_mutex: model_mutex,

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
        let mut learner_timer = PerformanceMonitor::new();
        global_timer.start();

        let mut model_ts = global_timer.get_duration();

        let speed_test = false;
        let mut speed_read = 0;

        let mut scanned_counter = 0;
        while num_iterations <= 0 || self.model.len() < num_iterations {
            if !speed_test {
                self.try_sample();
            }

            if self.learner.get_count() >= timeout {
                self.learner.shrink_target();
            }

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
                    // self.try_send_model();
                    true
                } else {
                    false
                };
            if found_new_rule {
                self.sum_gamma += self.learner.get_rho_gamma().powi(2);
                debug!(
                    "new-tree-info, {}, {}, {}, {}, {}",
                    self.model.len(),
                    self.learner.get_count(),
                    self.learner.get_rho_gamma(),
                    self.sum_gamma,
                    scanned_counter
                );
                scanned_counter = 0;
                model_ts = global_timer.get_duration();
                self.learner.reset();
                if interval > 0 && self.model.len() % interval == 0 {
                    self._validate();
                }
            }

            if self.handle_network() {
                model_ts = global_timer.get_duration();
            }
            if model_ts - self.last_backup_time >= 5.0 {
                self.handle_persistent(model_ts);
                self.last_backup_time = model_ts;
            }

            let (since_last_check, count, duration, speed) = global_timer.get_performance();
            if speed_test || since_last_check >= 2 {
                let (_, count_learn, duration_learn, speed_learn) = learner_timer.get_performance();
                debug!("boosting_speed, {}, {}, {}, {}, {}, {}, {}",
                       self.model.len(), duration, count, speed,
                       duration_learn, count_learn, speed_learn);
                global_timer.reset_last_check();

                speed_read += 1;
                if speed_test && speed_read >= 10 {
                    return;
                }
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
        if self.receiver.is_some() {
            // info!("Processing models received from the network");
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
                replaced = true;
                debug!("model-replaced, {}, {}, {}, {}",
                       self.sum_gamma, old_model_score, self.model.len(), old_model_size);
            } else {
                // info!("Remote models are not better. Skipped.");
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
        /*
        if let Ok(ref mut model) = self.model_mutex.try_lock() {
            **model = self.model.clone();
        }
        */
    }

    fn try_sample(&mut self) {
        /*
        let next_loader = if let Ok(loader) = self.next_training_loader.try_lock() {
            *loader
        } else {
            None
        };
        if next_loader.is_some() {
            let ess_option = self.training_loader.get_ess();
            let ess = if let Some(ess) = ess_option {
                debug!("training_sample_replaced, {}", ess);
            } else {
                debug!("training_sample_replaced, n/a");
            };
            self.training_loader = next_loader.unwrap();
            self.learner.reset_all();
        }
        */
    }

    fn _validate(&mut self) {
        info!("Validation is skipped.");
        // TODO: start validation in a non-blocking way
    }
}


fn get_base_tree(max_sample_size: usize, data_loader: &mut BufferLoader) -> (Tree, f32) {
    let mut remaining_reads = max_sample_size;
    let mut n_pos = 0;
    let mut n_neg = 0;
    while remaining_reads > 0 {
        data_loader.fetch_next_batch(true);
        let data = data_loader.get_curr_batch(true);
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
