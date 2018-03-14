use data_loader::DataLoader;
use learner::Learner;
use commons::Model;
use commons::LossFunc;

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
    model: Model
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
            model: vec![]
        };
        boosting.sample();
        boosting
    }

    pub fn training(
            &mut self,
            num_iterations: u32,
            max_trials_before_shrink: u32,
            validate_interval: u32) {
        debug!("Start training.");
        let interval = validate_interval as usize;
        let timeout = max_trials_before_shrink as usize;
        let mut remaining_iterations = num_iterations;
        while num_iterations <= 0 || remaining_iterations > 0 {
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
                match self.learner.get_new_weak_rule() {
                    &Some(ref weak_rule) => {
                        let tree = weak_rule.create_tree();
                        info!("A new tree is added: {:?}.", tree);
                        self.model.push(tree);
                        true
                    },
                    &None => {
                        false
                    }
                };
            if found_new_rule {
                if remaining_iterations > 0 {
                    remaining_iterations -= 1;
                }
                self.try_sample();
                self.learner.reset();
                if self.model.len() % interval == 0 {
                    self._validate();
                }
            }
        }
    }

    fn try_sample(&mut self) {
        let ess_option = self.training_loader_stack[1].get_ess();
        match ess_option {
            Some(ess) => {
                if ess < self.ess_threshold {
                    debug!("ESS is below the threshold: {} < {}. A new sample will be generated.",
                           ess, self.ess_threshold);
                    self.training_loader_stack.pop();
                    self.sample();
                }
            },
            None      => {}
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
