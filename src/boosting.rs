extern crate serde_json;

use rayon::prelude::*;

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
        let learner = Learner::new(default_rho_gamma, bins);

        // add root node for balancing labels
        let (base_tree, gamma) = get_base_tree(max_sample_size, &mut training_loader);
        let gamma_squared = gamma.powi(2);
        let model = vec![base_tree];

        let mut boosting = Boosting {
            training_loader_stack: vec![training_loader],
            testing_loader: testing_loader,
            eval_funcs: eval_funcs,

            sample_ratio: sample_ratio,
            ess_threshold: ess_threshold,

            learner: learner,
            model: model,

            sender: None,
            receiver: None,
            sum_gamma: gamma_squared.clone(),
            prev_sum_gamma: gamma_squared
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
            self.try_sample();
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
                    self.model.push(
                        weak_rule.create_tree()
                    );
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
        debug!("Checking network for a better model.");
        if self.receiver.is_some() {
            debug!("Processing models received from the network");
            // handle receiving
            let recv = self.receiver.as_ref().unwrap();
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
                self.model = best_model.unwrap();
                self.sum_gamma = max_score;
                self.prev_sum_gamma = self.sum_gamma;
                info!("A better model is received from the network: {} > {}. \
                      Local model size: {}. Remote model size: {}. \
                      Local model is replaced.",
                      max_score, self.sum_gamma, old_model_size, self.model.len());
            } else {
                debug!("Remote models are not better. Skipped.");
            }

            // handle sending
            if self.sum_gamma > self.prev_sum_gamma {
                self.sender.as_ref().unwrap().send((self.model.clone(), self.sum_gamma)).unwrap();
                self.prev_sum_gamma = self.sum_gamma;
            }
        } else {
            debug!("No new model received from the network");
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

    let prediction = 0.5 * (n_pos as f32 / n_neg as f32).ln();
    let mut tree = Tree::new(2);
    tree.split(0, 0, 0.0, prediction, prediction);
    tree.release();

    (tree, (0.5 - n_pos as f32 / (n_pos + n_neg) as f32).abs())
}
