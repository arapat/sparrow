use std::sync::mpsc;
use std::thread::sleep;
use std::thread::spawn;
use crossbeam_channel as channel;
use rayon::prelude::*;
use rand;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;
use self::channel::Sender;

use commons::ExampleWithScore;
use commons::Model;
use commons::performance_monitor::PerformanceMonitor;
use super::Strata;
use super::SharedCountsTable;
use super::SharedWeightsTable;

use commons::get_sign;
use commons::get_weight;


/// Sample examples from the stratified structure
///
/// Sampler read examples loaded from the stratified storage, and do
/// the following two tasks on these examples:
///
/// 1. update their scores,
/// 2. sample and send the sampled examples to the `BufferLoader` object.
///
/// The function uses the minimum variance sampling method to sample from
/// the examples loaded from the stratified storage.
///
/// * `sampled_examples`: the channle that the stratified storage sends examples loaded from disk.
/// * `updated_examples`: the channel that the sampler will send the examples after
/// updating their scores.

pub struct Samplers {
    counts_table: SharedCountsTable,
    weights_table: SharedWeightsTable,
    strata: Arc<RwLock<Strata>>,
    sampled_examples: mpsc::Sender<ExampleWithScore>,
    updated_examples: Sender<ExampleWithScore>,
    model: Arc<RwLock<Model>>,
}


impl Samplers {
    pub fn new(
        counts_table: SharedCountsTable,
        weights_table: SharedWeightsTable,
        strata: Arc<RwLock<Strata>>,
        sampled_examples: mpsc::Sender<ExampleWithScore>,
        updated_examples: Sender<ExampleWithScore>,
        next_model: mpsc::Receiver<Model>,
    ) -> Samplers {
        let model = Arc::new(RwLock::new(vec![]));
        {
            let model = model.clone();
            spawn(move || {
                for new_model in next_model.iter() {
                    let model_len = new_model.len();
                    let model = model.write();
                    *(model.unwrap()) = new_model;
                    info!("sampler model update, {}", model_len);
                }
            });
        }
        Samplers {
            counts_table: counts_table,
            weights_table: weights_table,
            strata: strata,
            sampled_examples: sampled_examples,
            updated_examples: updated_examples,
            model: model,
        }
    }

    pub fn run(&mut self, num_threads: usize) {
        for _ in 0..num_threads {
            let counts_table = self.counts_table.clone();
            let weights_table = self.weights_table.clone();
            let strata = self.strata.clone();
            let sampled_examples = self.sampled_examples.clone();
            let updated_examples = self.updated_examples.clone();
            let model = self.model.clone();
            spawn(move || {
                sampler(
                    counts_table, weights_table, strata, sampled_examples, updated_examples, model
                );
            });
        }
    }
}


fn sampler(
    counts_table: SharedCountsTable,
    weights_table: SharedWeightsTable,
    strata: Arc<RwLock<Strata>>,
    sampled_examples: mpsc::Sender<ExampleWithScore>,
    updated_examples: Sender<ExampleWithScore>,
    model: Arc<RwLock<Model>>,
) {
    let mut pm_update = PerformanceMonitor::new();
    let mut pm_sample = PerformanceMonitor::new();
    pm_update.start();
    pm_sample.start();

    let mut grids: HashMap<i8, f32> = HashMap::new();
    let mut killed = false;
    while !killed {
        // STEP 1: Sample which strata to get next sample
        let strata_weights: Vec<(i8, f64)> = {
            let mut hash_map = weights_table.read().unwrap();
            hash_map.iter()
                    .map(|(a, b)| (a.clone(), b.clone()))
                    .collect()
        };
        let index = get_sample_from(strata_weights);
        if index.is_none() {
            // stratified storage is empty, wait for data loading
            sleep(Duration::from_millis(1000));
            continue;
        }
        let index = index.unwrap();
        // STEP 2: Access the queue for the sampled strata
        let existing_receiver = {
            let read_strata = strata.read().unwrap();
            read_strata.get_out_queue(index)
        };
        let receiver = {
            if let Some(receiver) = existing_receiver {
                receiver
            } else {
                let mut strata = strata.write().unwrap();
                let (_, receiver) = strata.create(index);
                drop(strata);
                receiver
            }
        };
        // STEP 3: Sample one example using minimum variance sampling
        // meanwhile update the weights of all accessed examples
        let grid_size = 2f32.powi((index + 1) as i32);  // grid size can be set more conservatively
        let grid = grids.entry(index).or_insert(rand::random::<f32>() * grid_size);
        let mut removed_weights = 0.0;
        let mut removed_counts = 0;
        let mut sampled_example = None;
        while *grid < grid_size {
            let (example, (score, version)) = receiver.recv().unwrap();
            let (updated_score, model_size) = {
                let trees = model.read().unwrap();
                let model_size = trees.len();
                let inc_score: f32 = {
                    trees[version..model_size].par_iter().map(|tree| {
                        tree.get_leaf_prediction(&example)
                    }).sum()
                };
                (score + inc_score, model_size)
            };
            removed_counts += 1;
            removed_weights += get_weight(&example, score);
            *grid += get_weight(&example, updated_score);
            if *grid >= grid_size {
                sampled_example = Some((example.clone(), (updated_score, model_size)));
            }
            updated_examples.send((example, (updated_score, model_size)));
            pm_update.update(1);
        }
        // STEP 4: Correct the weights of this strata
        {
            let mut weights = weights_table.write().unwrap();
            let weight = weights.entry(index).or_insert(0.0);
            *weight -= removed_weights as f64;

            let mut counts = counts_table.write().unwrap();
            let count = counts.entry(index).or_insert(0);
            *count -= removed_counts;
        }
        // STEP 5: Send the sampled example to the buffer loader
        let sampled_example = sampled_example.unwrap();
        while *grid >= grid_size {
            if let Err(e) = sampled_examples.send(sampled_example.clone()) {
                error!("Sampler is killed because the sampled examples channel is closed: {}", e);
                killed = true;
                break;
            }
            *grid -= grid_size;
            pm_sample.update(1);
        }
        pm_update.write_log("sampler-update");
        pm_sample.write_log("sampler-sample");
    }
}


fn get_sample_from(p: Vec<(i8, f64)>) -> Option<i8> {
    let sum_of_weights: f64 = p.iter().map(|t| t.1).sum();
    if get_sign(sum_of_weights) == 0 {
        None
    } else {
        let mut frac = rand::random::<f64>() * sum_of_weights;
        let mut iter = p.iter();
        let mut key_val = &(0, 0.0);
        while get_sign(frac) >= 0 {
            key_val = iter.next().expect("get_sample_from: input p is empty");
            frac -= key_val.1;
        }
        Some(key_val.0)
    }
}