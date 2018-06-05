
use std::sync::Arc;
use std::sync::RwLock;
use chan::Receiver;

use std::thread::spawn;

use super::mpmc_map::MPMCMap;
use super::ExampleWithScore;
use super::WeightsTable;

use commons::get_weight;


type SharedMPMCMap = Arc<RwLock<MPMCMap>>;


pub struct Assigners {
    weights_table: WeightsTable,
    updated_examples_out: Receiver<ExampleWithScore>,
    in_queue: SharedMPMCMap
}


impl Assigners {
    pub fn new(weights_table: WeightsTable,
               updated_examples_out: Receiver<ExampleWithScore>,
               in_queue: SharedMPMCMap) -> Assigners {
        Assigners {
            weights_table: weights_table,
            updated_examples_out: updated_examples_out,
            in_queue: in_queue
        }
    }

    pub fn run(&mut self, num_threads: usize) {
        for _ in 0..num_threads {
            let weights_table = self.weights_table.clone(); 
            let updated_examples_out = self.updated_examples_out.clone();
            let in_queue = self.in_queue.clone();
            spawn(move|| {
                clear_updated_examples(weights_table, updated_examples_out, in_queue);
            });
        }
    }
}


fn clear_updated_examples(weights_table: WeightsTable,
                          updated_examples_out: Receiver<ExampleWithScore>,
                          in_queue: SharedMPMCMap) {
    while let Some(ret) = updated_examples_out.recv() {
        let (example, (score, version)) = ret;
        let weight = get_weight(&example, score);
        let index = get_strata_idx(weight);
        let (sender, _) = in_queue.write().unwrap().get(index);

        sender.send((example, (score, version)));
        let mut weights = weights_table.write().unwrap();
        let prev_weight = weights.entry(index).or_insert(0.0);
        *prev_weight += weight as f64;
    }
    error!("Updated examples queue was closed.");
}


fn get_strata_idx(weight: f32) -> i8 {
    weight.log2() as i8
}