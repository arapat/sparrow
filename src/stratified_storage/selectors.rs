use std::sync::Arc;
use std::sync::RwLock;
use chan::Sender;

use rand;
use std::thread::spawn;

use super::ExampleWithScore;
use super::Strata;
use super::WeightsTable;

use commons::get_sign;


pub struct Selectors {
    weights_table: WeightsTable,
    strata: Arc<RwLock<Strata>>,
    loaded_examples: Sender<ExampleWithScore>
}


impl Selectors {
    pub fn new(weights_table: WeightsTable,
               strata: Arc<RwLock<Strata>>,
               loaded_examples: Sender<ExampleWithScore>) -> Selectors {
        Selectors {
            weights_table: weights_table,
            strata: strata,
            loaded_examples: loaded_examples
        }
    }

    pub fn run(&mut self, num_threads: usize) {
        for _ in 0..num_threads {
            let weights_table = self.weights_table.clone();
            let strata = self.strata.clone();
            let loaded_examples = self.loaded_examples.clone();
            spawn(move || {
                fill_loaded_examples(weights_table, strata, loaded_examples);
            });
        }
    }
}


fn fill_loaded_examples(weights_table: WeightsTable,
                        strata: Arc<RwLock<Strata>>,
                        loaded_examples: Sender<ExampleWithScore>) {
    loop {
        let sample = {
            let mut hash_map = weights_table.write().unwrap();
            let p: Vec<(i8, f64)> = hash_map.iter()
                                            .map(|(a, b)| (a.clone(), b.clone()))
                                            .collect();
            if let Some(index) = get_sample_from(p) {
                let receiver = {
                    if let Some(t) = strata.read().unwrap().get_out_queue(index) {
                        t
                    } else {
                        let (_, receiver) = strata.write().unwrap().create(index);
                        receiver
                    }
                };
                let example = receiver.recv().unwrap();
                let weight = hash_map.entry(index).or_insert(0.0);
                *weight -= (example.1).0 as f64;
                Some(example)
            } else {
                None
            }
        };
        if let Some(example) = sample {
            loaded_examples.send(example);
        }
    }
}


fn get_sample_from(p: Vec<(i8, f64)>) -> Option<i8> {
    let sum_of_weights: f64 = p.iter().map(|t| t.1).sum();
    let mut frac = rand::random::<f64>() * sum_of_weights;
    for (key, val) in p.iter() {
        frac -= val;
        if get_sign(frac) <= 0 {
            return Some(*key);
        }
    }
    None
}