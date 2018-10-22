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
use super::SPEED_TEST;

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
    sampled_examples: Sender<(ExampleWithScore, u32)>,
    updated_examples: Sender<ExampleWithScore>,
    model: Arc<RwLock<Model>>,
    stats_update_s: Sender<(i8, (i32, f64))>,
    weights_table: WeightTableRead,
}


impl Samplers {
    pub fn new(
        strata: Arc<RwLock<Strata>>,
        sampled_examples: Sender<(ExampleWithScore, u32)>,
        updated_examples: Sender<ExampleWithScore>,
        next_model: channel::Receiver<Model>,
        stats_update_s: Sender<(i8, (i32, f64))>,
        weights_table: WeightTableRead,
    ) -> Samplers {
        let model = Arc::new(RwLock::new(vec![]));
        {
            let model = model.clone();
            spawn(move || {
                while let Some(new_model) = next_model.recv() {
                    let model_len = new_model.len();
                    {
                        let mut model = model.write().unwrap();
                        *model = new_model;
                    }
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
    sampled_examples: Sender<(ExampleWithScore, u32)>,
    updated_examples: Sender<ExampleWithScore>,
    model: Arc<RwLock<Model>>,
    stats_update_s: Sender<(i8, (i32, f64))>,
    weights_table: WeightTableRead,
) {
    let mut pm_update = PerformanceMonitor::new();
    let mut pm_sample = PerformanceMonitor::new();
    pm_update.start();
    pm_sample.start();

    let mut grids: HashMap<i8, f32> = HashMap::new();

    let mut pm_total = PerformanceMonitor::new();
    let mut pm1 = PerformanceMonitor::new();
    let mut pm2 = PerformanceMonitor::new();
    let mut pm3 = PerformanceMonitor::new();
    let mut pm4 = PerformanceMonitor::new();
    pm_total.start();

    loop {
        // STEP 1: Sample which strata to get next sample
        pm1.resume();
        let index = super::sample_weights_table(&weights_table);
        if index.is_none() {
            // stratified storage is empty, wait for data loading
            sleep(Duration::from_millis(1000));
            pm1.pause();
            continue;
        }
        let index = index.unwrap();
        pm1.pause();
        // STEP 2: Access the queue for the sampled strata
        pm2.resume();
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
                receiver
            }
        };
        pm2.pause();
        // STEP 3: Sample one example using minimum variance sampling
        // meanwhile update the weights of all accessed examples
        pm3.resume();
        let grid_size = {
            if SPEED_TEST {
                // assume sampling 1% of the weight sequence 1, 2, ..., 10
                55.0 * 10.0
            } else {
                // grid size can be set more conservatively
                2f32.powi((index + 1) as i32)
            }
        };
        let grid = grids.entry(index).or_insert(rand::random::<f32>() * grid_size);
        let mut sampled_example = None;
        while sampled_example.is_none() {
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
            stats_update_s.send((index, (-1, -get_weight(&example, score) as f64)));
            updated_examples.send((example, (updated_score, model_size)));
            pm_update.update(1);
        }
        pm3.pause();
        // STEP 4: Send the sampled example to the buffer loader
        pm4.resume();
        let sampled_example = sampled_example.unwrap();
        let sample_count = (*grid / grid_size) as u32;
        if sample_count > 0 {
            sampled_examples.send((sampled_example, sample_count));
            *grid -= grid_size * (sample_count as f32);
            pm_sample.update(sample_count as usize);
        }
        pm4.pause();

        pm_update.write_log("sampler-update");
        pm_sample.write_log("sampler-sample");
        if pm_total.get_duration() > 5.0 {
            let total = pm_total.get_duration();
            let step1 = pm1.get_duration() / total;
            let step2 = pm2.get_duration() / total;
            let step3 = pm3.get_duration() / total;
            let step4 = pm4.get_duration() / total;
            let others = 1.0 - step1 - step2 - step3 - step4;
            pm_total.reset();
            pm1.reset();
            pm2.reset();
            pm3.reset();
            pm4.reset();
            debug!("sample-perf-breakdown, {}, {}, {}, {}, {}", step1, step2, step3, step4, others);
        }
    }
}
