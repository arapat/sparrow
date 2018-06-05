use rand;

use std::sync::Arc;
use std::sync::RwLock;
use chan::Sender;
use chan::Receiver;

use std::thread::spawn;

use commons::Model;
use commons::ExampleWithScore;

use commons::get_sign;
use commons::get_weight;


pub struct Sampler {
    loaded_examples: Receiver<ExampleWithScore>,
    updated_examples: Sender<ExampleWithScore>,
    sampled_examples: Sender<ExampleWithScore>,

    next_model: Receiver<Model>,
    model: Arc<RwLock<Model>>
}


impl Sampler {
    pub fn new(loaded_examples: Receiver<ExampleWithScore>,
               updated_examples: Sender<ExampleWithScore>,
               sampled_examples: Sender<ExampleWithScore>,
               next_model: Receiver<Model>) -> Sampler {
        Sampler {
            loaded_examples: loaded_examples,
            updated_examples: updated_examples,
            sampled_examples: sampled_examples,

            next_model: next_model,
            model: Arc::new(RwLock::new(vec![]))
        }
    }

    pub fn run(&self, num_threads: usize, initial_grid_size: f32, average_window_size: usize) {
        self.run_model_updates();

        for _ in 0..num_threads {
            let grid_size = initial_grid_size.clone();
            let window_size = average_window_size.clone();
            let loaded_examples = self.loaded_examples.clone();
            let updated_examples = self.updated_examples.clone();
            let sampled_examples = self.sampled_examples.clone();
            let model = self.model.clone();
            spawn(move|| {
                sample(grid_size, window_size, loaded_examples,
                       updated_examples, sampled_examples, model);
            });
        }
    }

    fn run_model_updates(&self) {
        let model = self.model.clone();
        let next_model = self.next_model.clone();
        spawn(move|| {
            update_model(model, next_model);
        });
    }
}


fn update_model(model: Arc<RwLock<Model>>, next_model: Receiver<Model>) {
    for new_model in next_model.iter() {
        let mut model_w = model.write().unwrap();
        *model_w = new_model;
    }
}


fn sample(initial_grid_size: f32,
          window_size: usize,
          loaded_examples: Receiver<ExampleWithScore>,
          updated_examples: Sender<ExampleWithScore>,
          sampled_examples: Sender<ExampleWithScore>,
          model: Arc<RwLock<Model>>) {
    let mut grid_size = initial_grid_size;
    let mut grid = -rand::random::<f32>() * grid_size;
    let rou = 1.0 / window_size as f32;
    loop {
        let (data, (score, base_node)) = loaded_examples.recv().expect(
            "Loaded example queue is closed."
        );
        let trees = model.read().unwrap();
        let model_size = trees.len();
        let mut updated_score = score;
        for tree in trees[base_node..model_size].iter() {
            updated_score += tree.get_leaf_prediction(&data);
        }
        updated_examples.send((data.clone(), (updated_score, model_size)));

        let w = get_weight(&data, updated_score);
        grid_size = grid_size * (1.0 - rou) + w * rou;
        grid += w;
        while get_sign(grid as f64) >= 0 {
            sampled_examples.send((data.clone(), (updated_score, model_size)));
            grid -= grid_size;
        }
    }
}