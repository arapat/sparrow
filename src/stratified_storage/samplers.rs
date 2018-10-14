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
use super::WeightTableRead;

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
    strata: Arc<RwLock<Strata>>,
    sampled_examples: mpsc::Sender<ExampleWithScore>,
    updated_examples: Sender<ExampleWithScore>,
    model: Arc<RwLock<Model>>,
    stats_update_s: mpsc::Sender<(i8, (i32, f64))>,
    weights_table: WeightTableRead,
}


impl Samplers {
    pub fn new(
        strata: Arc<RwLock<Strata>>,
        sampled_examples: mpsc::Sender<ExampleWithScore>,
        updated_examples: Sender<ExampleWithScore>,
        next_model: mpsc::Receiver<Model>,
        stats_update_s: mpsc::Sender<(i8, (i32, f64))>,
        weights_table: WeightTableRead,
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
            strata: strata,
            sampled_examples: sampled_examples,
            updated_examples: updated_examples,
            model: model,
            stats_update_s: stats_update_s,
            weights_table: weights_table,
        }
    }

    pub fn run(&mut self, num_threads: usize) {
        for _ in 0..num_threads {
            let strata = self.strata.clone();
            let sampled_examples = self.sampled_examples.clone();
            let updated_examples = self.updated_examples.clone();
            let model = self.model.clone();
            let stats_update_s = self.stats_update_s.clone();
            let weights_table = self.weights_table.clone();
            spawn(move || {
                sampler(
                    strata, sampled_examples, updated_examples, model, stats_update_s, weights_table
                );
            });
        }
    }
}


fn sampler(
    strata: Arc<RwLock<Strata>>,
    sampled_examples: mpsc::Sender<ExampleWithScore>,
    updated_examples: Sender<ExampleWithScore>,
    model: Arc<RwLock<Model>>,
    stats_update_s: mpsc::Sender<(i8, (i32, f64))>,
    weights_table: WeightTableRead,
) {
    let mut pm_update = PerformanceMonitor::new();
    let mut pm_sample = PerformanceMonitor::new();
    pm_update.start();
    pm_sample.start();

    let mut grids: HashMap<i8, f32> = HashMap::new();
    let mut killed = false;
    while !killed {
        // STEP 1: Sample which strata to get next sample
        let index = super::sample_weights_table(&weights_table);
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
            *grid += get_weight(&example, updated_score);
            if *grid >= grid_size {
                sampled_example = Some((example.clone(), (updated_score, model_size)));
            }
            stats_update_s.send((index, (-1, -get_weight(&example, score) as f64))).unwrap();
            updated_examples.send((example, (updated_score, model_size)));
            pm_update.update(1);
        }
        // STEP 4: Send the sampled example to the buffer loader
        let sampled_example = sampled_example.unwrap();
        while *grid >= grid_size {
            if let Err(_) = sampled_examples.send(sampled_example.clone()) {
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
