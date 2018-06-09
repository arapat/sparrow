
use std::sync::Arc;
use std::sync::RwLock;
use chan::Receiver;

use std::thread::spawn;

use super::ExampleWithScore;
use super::Strata;
use super::WeightsTable;

use commons::get_weight;


pub struct Assigners {
    weights_table: WeightsTable,
    updated_examples_out: Receiver<ExampleWithScore>,
    strata: Arc<RwLock<Strata>>
}


impl Assigners {
    pub fn new(weights_table: WeightsTable,
               updated_examples_out: Receiver<ExampleWithScore>,
               strata: Arc<RwLock<Strata>>) -> Assigners {
        Assigners {
            weights_table: weights_table,
            updated_examples_out: updated_examples_out,
            strata: strata
        }
    }

    pub fn run(&mut self, num_threads: usize) {
        for _ in 0..num_threads {
            let weights_table = self.weights_table.clone(); 
            let updated_examples_out = self.updated_examples_out.clone();
            let strata = self.strata.clone();
            spawn(move|| {
                clear_updated_examples(weights_table, updated_examples_out, strata);
            });
        }
    }
}


fn clear_updated_examples(weights_table: WeightsTable,
                          updated_examples_out: Receiver<ExampleWithScore>,
                          strata: Arc<RwLock<Strata>>) {
    while let Some(ret) = updated_examples_out.recv() {
        let (example, (score, version)) = ret;
        let weight = get_weight(&example, score);
        let index = get_strata_idx(weight);
        let mut sender = {
            if let Some(t) = strata.read().unwrap().get_in_queue(index) {
                t
            } else {
                let (sender, _) = strata.write().unwrap().create(index);
                sender
            }
        };
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