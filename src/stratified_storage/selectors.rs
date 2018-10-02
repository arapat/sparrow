use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;
use chan::Sender;

use rand;
use std::thread::sleep;
use std::thread::spawn;

use commons::ExampleWithScore;
use commons::performance_monitor::PerformanceMonitor;
use super::Strata;
use super::CountsTable;
use super::WeightsTable;

use commons::get_sign;
use commons::get_weight;


pub struct Selectors {
    counts_table: CountsTable,
    weights_table: WeightsTable,
    strata: Arc<RwLock<Strata>>,
    loaded_examples: Sender<ExampleWithScore>
}


impl Selectors {
    pub fn new(counts_table: CountsTable,
               weights_table: WeightsTable,
               strata: Arc<RwLock<Strata>>,
               loaded_examples: Sender<ExampleWithScore>) -> Selectors {
        Selectors {
            counts_table: counts_table,
            weights_table: weights_table,
            strata: strata,
            loaded_examples: loaded_examples
        }
    }

    pub fn run(&mut self, num_threads: usize) {
        for _ in 0..num_threads {
            let counts_table = self.counts_table.clone();
            let weights_table = self.weights_table.clone();
            let strata = self.strata.clone();
            let loaded_examples = self.loaded_examples.clone();
            spawn(move || {
                fill_loaded_examples(counts_table, weights_table, strata, loaded_examples);
            });
        }
    }
}


fn fill_loaded_examples(counts_table: CountsTable,
                        weights_table: WeightsTable,
                        strata: Arc<RwLock<Strata>>,
                        loaded_examples: Sender<ExampleWithScore>) {
    let mut pm = PerformanceMonitor::new();
    pm.start();
    loop {
        let p: Vec<(i8, f64)> = {
            let mut hash_map = weights_table.read().unwrap();
            hash_map.iter()
                    .map(|(a, b)| (a.clone(), b.clone()))
                    .collect()
        };
        if let Some(index) = get_sample_from(p) {
            let read_strata = strata.read().unwrap();
            let existing_receiver = read_strata.get_out_queue(index);
            drop(read_strata);
            let receiver = {
                if let Some(receiver) = existing_receiver {
                    receiver
                } else {
                    let (_, receiver) = strata.write().unwrap().create(index);
                    receiver
                }
            };
            let (example, (score, version)) = receiver.recv().unwrap();
            let weight = get_weight(&example, score);
            {
                let mut weights = weights_table.write().unwrap();
                let prev_weight = weights.entry(index).or_insert(0.0);
                *prev_weight -= weight as f64;

                let mut counts = counts_table.write().unwrap();
                let prev_count = counts.entry(index).or_insert(0);
                *prev_count -= 1;
            }
            loaded_examples.send((example, (score, version)));
            pm.update(1);
            pm.write_log("selector");
        } else {
            sleep(Duration::from_millis(100));
        }
    }
}


fn get_sample_from(p: Vec<(i8, f64)>) -> Option<i8> {
    let sum_of_weights: f64 = p.iter().map(|t| t.1).sum();
    let mut frac = rand::random::<f64>() * sum_of_weights;
    for (key, val) in p.iter() {
        frac -= val;
        if get_sign(frac) < 0 {
            return Some(*key);
        }
    }
    None
}