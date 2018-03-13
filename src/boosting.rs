use data_loader::DataLoader;
use learner::Learner;
use commons::Model;
use commons::LossFunc;

use commons::get_weights;
use bins::create_bins;
use validator::validate;


pub struct Boosting<'a> {
    training_loader: DataLoader,
    testing_loader: DataLoader,
    eval_funcs: Vec<&'a LossFunc>,

    learner: Learner,
    model: Model
}

impl<'a> Boosting<'a> {
    pub fn new(
                mut training_loader: DataLoader, testing_loader: DataLoader,
                max_sample_size: usize, max_bin_size: usize,
                default_rho_gamma: f32,
                eval_funcs: Vec<&'a LossFunc>
            ) -> Boosting<'a> {
        let bins = create_bins(max_sample_size, max_bin_size, &mut training_loader);
        let learner = Learner::new(training_loader.get_feature_size(), default_rho_gamma, bins);
        Boosting {
            training_loader: training_loader,
            testing_loader: testing_loader,
            eval_funcs: eval_funcs,
            learner: learner,
            model: vec![]
        }
    }

    pub fn training(&mut self, num_iterations: u32, max_trials: u32, validate_interval: u32) {
        let interval = validate_interval as usize;
        let timeout = max_trials as usize;
        let mut remaining_iterations = num_iterations;
        while remaining_iterations > 0 {
            if self.learner.get_count() >= timeout {
                self.learner.shrink_target();
            }

            {
                self.training_loader.fetch_next_batch();
                self.training_loader.fetch_scores(&self.model);
                let data = self.training_loader.get_curr_batch();
                let scores = self.training_loader.get_relative_scores();
                let weights = get_weights(data, scores.as_slice());
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
                remaining_iterations -= 1;
                self.learner.reset();
                if self.model.len() % interval == 0 {
                    self._validate();
                }
            }
        }
    }

    fn _validate(&mut self) {
        debug!("Validation is started.");
        let scores = validate(&mut self.testing_loader, &self.model, &self.eval_funcs);
        let output: Vec<String> = scores.into_iter().map(|x| x.to_string()).collect();
        info!("Eval funcs: {}", output.join(", "));
        debug!("Validation is completed.");
    }
}
