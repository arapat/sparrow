extern crate serde_json;

use std::sync::mpsc;
use std::sync::mpsc::Sender;
use std::sync::mpsc::Receiver;

use data_loader::DataLoader;
use learner::Learner;
use network::start_network;
use commons::Model;
use commons::LossFunc;
use commons::PerformanceMonitor;
use commons::ModelScore;

use commons::get_weights;
use bins::create_bins;
use validator::validate;


pub struct Boosting<'a> {
    training_loader_stack: Vec<DataLoader>,
    testing_loader: DataLoader,
    eval_funcs: Vec<&'a LossFunc>,

    sample_ratio: f32,
    ess_threshold: f32,

    learner: Learner,
    model: Model,

    sender: Option<Sender<ModelScore>>,
    receiver: Option<Receiver<ModelScore>>,
    sum_gamma: f32,
    prev_sum_gamma: f32
}

impl<'a> Boosting<'a> {
    pub fn new(
                mut training_loader: DataLoader,
                testing_loader: DataLoader,
                max_sample_size: usize,
                max_bin_size: usize,
                sample_ratio: f32,
                ess_threshold: f32,
                default_rho_gamma: f32,
                eval_funcs: Vec<&'a LossFunc>
            ) -> Boosting<'a> {
        let bins = create_bins(max_sample_size, max_bin_size, &mut training_loader);
        let learner = Learner::new(training_loader.get_feature_size(), default_rho_gamma, bins);
        let mut boosting = Boosting {
            training_loader_stack: vec![training_loader],
            testing_loader: testing_loader,
            eval_funcs: eval_funcs,

            sample_ratio: sample_ratio,
            ess_threshold: ess_threshold,

            learner: learner,
            model: vec![],

            sender: None,
            receiver: None,
            sum_gamma: 0.0,
            prev_sum_gamma: 0.0
        };
        boosting.sample();
        boosting
    }

    pub fn enable_network(&mut self, remote_ips: &Vec<String>, port: u16) {
        let (local_send, local_recv): (Sender<ModelScore>, Receiver<ModelScore>) = mpsc::channel();
        let (other_send, other_recv): (Sender<ModelScore>, Receiver<ModelScore>) = mpsc::channel();
        self.sender = Some(local_send);
        self.receiver = Some(other_recv);
        start_network(remote_ips, port, other_send, local_recv)
    }

    pub fn training(
            &mut self,
            num_iterations: u32,
            max_trials_before_shrink: u32,
            validate_interval: u32) {
        debug!("Start training.");
        let interval = validate_interval as usize;
        let timeout = max_trials_before_shrink as usize;
        let mut iteration = 0;
        let timer = PerformanceMonitor::new();
        while num_iterations <= 0 || iteration < num_iterations {
            if self.learner.get_count() >= timeout {
                self.learner.shrink_target();
            }

            {
                let training_loader = &mut self.training_loader_stack[1];
                training_loader.fetch_next_batch();
                training_loader.fetch_scores(&self.model);
                let data = training_loader.get_curr_batch();
                let scores = training_loader.get_relative_scores();
                let weights = get_weights(data, scores);
                self.learner.update(data, &weights);
            }

            let found_new_rule =
                if let &Some(ref weak_rule) = self.learner.get_new_weak_rule() {
                    let tree = weak_rule.create_tree();
                    self.model.push(tree);
                    true
                } else {
                    false
                };
            if found_new_rule {
                info!(
                    "Clock {}. Tree {} is added: {:?}. Scanned {} examples. Advantage is {}.",
                    timer.get_duration(), iteration, self.model[iteration as usize],
                    self.learner.get_count(), self.learner.get_rho_gamma()
                );
                iteration += 1;
                self.sum_gamma += self.learner.get_rho_gamma().powi(2);
                self.try_sample();
                self.learner.reset();
                if self.model.len() % interval == 0 {
                    self._validate();
                }
            }

            self.handle_network();
        }
        info!("Model = {}", serde_json::to_string(&self.model).unwrap());
    }

    fn handle_network(&mut self) {
        if self.receiver.is_some() {
            // handle receiving
            let recv = self.receiver.as_ref().unwrap();
            let mut best_model = None;
            let mut max_score = 0.0;
            while let Some(model_score) = recv.try_iter().next() {
                let (new_model, sum_gamma) = model_score;
                if best_model.is_none() || sum_gamma > max_score {
                    best_model = Some(new_model);
                    max_score = sum_gamma;
                }
            }
            if max_score > self.sum_gamma {
                self.model = best_model.unwrap();
                self.sum_gamma = max_score;
                self.prev_sum_gamma = self.sum_gamma;
            }

            // handle sending
            if self.sum_gamma > self.prev_sum_gamma {
                self.sender.as_ref().unwrap().send((self.model.clone(), self.sum_gamma)).unwrap();
                self.prev_sum_gamma = self.sum_gamma;
            }
        }
    }

    fn try_sample(&mut self) {
        let ess_option = self.training_loader_stack[1].get_ess();
        if let Some(ess) = ess_option {
            if ess < self.ess_threshold {
                debug!("ESS is below the threshold: {} < {}. A new sample will be generated.",
                        ess, self.ess_threshold);
                self.training_loader_stack.pop();
                self.sample();
                self.learner.reset_all();
            }
        }
    }

    fn sample(&mut self) {
        info!("Re-sampling is started.");
        let new_sample = self.training_loader_stack[0].sample(&self.model, self.sample_ratio);
        self.training_loader_stack.push(new_sample);
        info!("A new sample is generated.");
    }

    fn _validate(&mut self) {
        debug!("Validation is started.");
        let scores = validate(&mut self.testing_loader, &self.model, &self.eval_funcs);
        let output: Vec<String> = scores.into_iter().map(|x| x.to_string()).collect();
        info!("Eval funcs: {}", output.join(", "));
        debug!("Validation is completed.");
    }
}
