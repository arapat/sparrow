use std::thread::sleep;
use std::thread::spawn;
use rand;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

use commons::channel::Sender;
use commons::performance_monitor::PerformanceMonitor;
use commons::ExampleWithScore;
use commons::model::Model;
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
    sampled_examples: Sender<((ExampleWithScore, u32), u32)>,
    updated_examples: Sender<ExampleWithScore>,
    model: Arc<RwLock<Model>>,
    stats_update_s: Sender<(i8, (i32, f64))>,
    weights_table: WeightTableRead,
    num_threads: usize,
    sampler_state: Arc<RwLock<bool>>,
}


// TODO: potential bug, sample function may miss the stop signal and go directly to a new
// start signal, in which case we would loss control of the total number of sampler threads
impl Samplers {
    pub fn new(
        model: Arc<RwLock<Model>>,
        strata: Arc<RwLock<Strata>>,
        sampled_examples: Sender<((ExampleWithScore, u32), u32)>,
        updated_examples: Sender<ExampleWithScore>,
        stats_update_s: Sender<(i8, (i32, f64))>,
        weights_table: WeightTableRead,
        num_threads: usize,
        sampler_state: Arc<RwLock<bool>>,
    ) -> Samplers {
        Samplers {
            strata: strata,
            sampled_examples: sampled_examples,
            updated_examples: updated_examples,
            model: model,
            stats_update_s: stats_update_s,
            weights_table: weights_table,
            num_threads: num_threads,
            sampler_state: sampler_state,
        }
    }

    pub fn run(&self, stratified_size: Arc<RwLock<usize>>, capacity: usize) {
        let num_threads      = self.num_threads;
        let strata           = self.strata.clone();
        let sampled_examples = self.sampled_examples.clone();
        let updated_examples = self.updated_examples.clone();
        let model            = self.model.clone();
        let stats_update_s   = self.stats_update_s.clone();
        let weights_table    = self.weights_table.clone();
        let sampler_state    = self.sampler_state.clone();
        for _ in 0..num_threads {
            let strata           = strata.clone();
            let sampled_examples = sampled_examples.clone();
            let updated_examples = updated_examples.clone();
            let model            = model.clone();
            let stats_update_s   = stats_update_s.clone();
            let weights_table    = weights_table.clone();
            let sampler_state    = sampler_state.clone();
            let size             = stratified_size.clone();
            spawn(move || {
                let mut is_enough_loaded = false;
                while is_enough_loaded {
                    let size = size.read().unwrap();
                    is_enough_loaded = *size > capacity / 3;
                    drop(size);
                }
                sampler(
                    strata, sampled_examples, updated_examples,
                    model, stats_update_s, weights_table, sampler_state,
                );
            });
        }
    }
}


fn sampler(
    strata: Arc<RwLock<Strata>>,
    sampled_examples: Sender<((ExampleWithScore, u32), u32)>,
    updated_examples: Sender<ExampleWithScore>,
    model: Arc<RwLock<Model>>,
    stats_update_s: Sender<(i8, (i32, f64))>,
    weights_table: WeightTableRead,
    sampler_state: Arc<RwLock<bool>>,
) {
    let mut pm_update = PerformanceMonitor::new();
    let mut pm_sample = PerformanceMonitor::new();
    pm_update.start();
    pm_sample.start();

    let mut grids: HashMap<i8, f32> = HashMap::new();

    let mut num_scanned = 0;
    let mut _num_sampled = 0;

    loop {
        pm_update.write_log("sampler-update");
        pm_sample.write_log("sampler-sample");

        // STEP 1: Sample which strata to get next sample
        let index = super::sample_weights_table(&weights_table);
        if index.is_none() {
            // stratified storage is empty, wait for data loading
            debug!("sampler, Sampler sleeps waiting for data loading");
            sleep(Duration::from_millis(1000));
            continue;
        }
        let index = index.unwrap();
        // STEP 2: Access the queue for the sampled strata
        let existing_receiver = {
            let read_strata = strata.try_read();
            if read_strata.is_err() {
                debug!("sampler, sampler cannot read the strata, {}", index);
                continue;
            }
            read_strata.unwrap().get_out_queue(index)
        };
        let receiver = existing_receiver.unwrap();
        // STEP 3: Sample one example using minimum variance sampling
        // meanwhile update the weights of all accessed examples
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
        let mut sampled_trials = 0;
        let mut retrieve_pm = PerformanceMonitor::new();
        let mut last_error_timer = retrieve_pm.get_duration();
        retrieve_pm.start();
        while sampled_example.is_none() {
            sampled_trials += 1;
            let recv = {
                let mut failed_recv = 0;
                let mut recv = receiver.try_recv();
                while recv.is_none() && failed_recv < 5 {
                    failed_recv += 1;
                    sleep(Duration::from_millis(500));
                    recv = receiver.try_recv();
                }
                recv
            };
            if recv.is_none() {
                break;
            }
            let (example, (score, version)) = recv.unwrap();
            let (updated_score, model_size) = {
                let latest_model = &model.read().unwrap();
                let (inc_score, (model_size, _)) = latest_model.get_prediction(&example, version);
                (score + inc_score, model_size)
            };
            let updated_weight = get_weight(&example, updated_score);
            if updated_weight.log2() as i8 == index {
                *grid += updated_weight;
                if *grid >= grid_size {
                    sampled_example = Some((example.clone(), (updated_score, model_size)));
                }
            }
            num_scanned += 1;
            stats_update_s.send((index, (-1, -get_weight(&example, score) as f64)));
            updated_examples.send((example, (updated_score, model_size)));
            pm_update.update(1);
            if retrieve_pm.get_duration() - last_error_timer >= 10.0 {
                debug!("sampler, rejection high in one stratum, {}, {}, {}, {}",
                            index, *grid, grid_size, sampled_trials);
                last_error_timer = retrieve_pm.get_duration();
            }
        }
        // STEP 4: Send the sampled example to the buffer loader
        if sampled_example.is_none() {
            debug!("Sampling the stratum {} failed because it has too few examples", index);
            continue;
        }
        let sampled_example = sampled_example.unwrap();
        let sample_count = (*grid / grid_size) as u32;
        if sample_count > 0 {
            sampled_examples.send(((sampled_example, sample_count), num_scanned));
            *grid -= grid_size * (sample_count as f32);
            _num_sampled += sample_count;
            num_scanned = 0;
            pm_sample.update(sample_count as usize);
        }

        // terminate if sampler is stopping
        {
            let sampler_state = sampler_state.read().unwrap();
            if *sampler_state == false {
                break;
            }
        }
    }
}
