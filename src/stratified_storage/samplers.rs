use std::thread::sleep;
use std::thread::spawn;
use rand;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

use commons::channel::Sender;
use commons::channel::Receiver;
use commons::performance_monitor::PerformanceMonitor;
use commons::ExampleWithScore;
use commons::Model;
use commons::Signal;
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
    next_model: Receiver<Model>,
    model: Arc<RwLock<Model>>,
    sampling_signal: Arc<RwLock<Signal>>,
    stats_update_s: Sender<(i8, (i32, f64))>,
    weights_table: WeightTableRead,
    // sampling_signal_channel: Receiver<Signal>,
    num_threads: usize,
}


// TODO: potential bug, sample function may miss the stop signal and go directly to a new
// start signal, in which case we would loss control of the total number of sampler threads
impl Samplers {
    pub fn new(
        strata: Arc<RwLock<Strata>>,
        sampled_examples: Sender<(ExampleWithScore, u32)>,
        updated_examples: Sender<ExampleWithScore>,
        next_model: Receiver<Model>,
        stats_update_s: Sender<(i8, (i32, f64))>,
        weights_table: WeightTableRead,
        _sampling_signal_channel: Receiver<Signal>,
        num_threads: usize,
    ) -> Samplers {
        Samplers {
            strata: strata,
            sampled_examples: sampled_examples,
            updated_examples: updated_examples,
            next_model: next_model,
            model: Arc::new(RwLock::new(vec![])),
            sampling_signal: Arc::new(RwLock::new(Signal::STOP)),
            stats_update_s: stats_update_s,
            weights_table: weights_table,
            // sampling_signal_channel: sampling_signal_channel,
            num_threads: num_threads,
        }
    }

    pub fn run(&self) {
        let next_model = self.next_model.clone();
        let model = self.model.clone();
        spawn(move || {
            while let Some(new_model) = next_model.recv() {
                let model_len = new_model.len();
                {
                    let mut model = model.write().unwrap();
                    *model = new_model;
                }
                debug!("sampler model update, {}", model_len);
            }
        });

        // let signal_channel   = self.sampling_signal_channel.clone();
        let num_threads      = self.num_threads;
        let strata           = self.strata.clone();
        let sampled_examples = self.sampled_examples.clone();
        let updated_examples = self.updated_examples.clone();
        let model            = self.model.clone();
        let sampling_signal  = self.sampling_signal.clone();
        let stats_update_s   = self.stats_update_s.clone();
        let weights_table    = self.weights_table.clone();
        // TODO: This should be a feature only enable in the debugging mode
        // spawn(move || {
        //     while let Some(new_signal) = signal_channel.recv() {
                // debug!("updating sampling signal, {:?}", new_signal);
                let start = {
                    let mut signal = sampling_signal.write().unwrap();
                    // assert!(*signal != new_signal);
                    *signal = Signal::START;  // new_signal;
                    *signal == Signal::START
                };
                if start {
                    for _ in 0..num_threads {
                        let strata           = strata.clone();
                        let sampled_examples = sampled_examples.clone();
                        let updated_examples = updated_examples.clone();
                        let model            = model.clone();
                        let sampling_signal  = sampling_signal.clone();
                        let stats_update_s   = stats_update_s.clone();
                        let weights_table    = weights_table.clone();
                        spawn(move || {
                            sampler(
                                strata, sampled_examples, updated_examples,
                                model, sampling_signal, stats_update_s, weights_table,
                            );
                        });
                    }
                }
        //     }
        // });
    }
}


fn sampler(
    strata: Arc<RwLock<Strata>>,
    sampled_examples: Sender<(ExampleWithScore, u32)>,
    updated_examples: Sender<ExampleWithScore>,
    model: Arc<RwLock<Model>>,
    // TODO: remove sampling signal if serial sampling is removed
    _sampling_signal: Arc<RwLock<Signal>>,
    stats_update_s: Sender<(i8, (i32, f64))>,
    weights_table: WeightTableRead,
) {
    let mut pm_update = PerformanceMonitor::new();
    let mut pm_sample = PerformanceMonitor::new();
    pm_update.start();
    pm_sample.start();

    let mut grids: HashMap<i8, f32> = HashMap::new();

    let mut pm_total = PerformanceMonitor::new();
    let mut pm3_total = PerformanceMonitor::new();
    let mut pm1 = PerformanceMonitor::new();
    let mut pm2 = PerformanceMonitor::new();
    let mut pm3 = PerformanceMonitor::new();
    let mut pm4 = PerformanceMonitor::new();
    let mut num_updated = 0;
    let mut num_sampled = 0;
    pm_total.start();

    loop {
        pm_update.write_log("sampler-update");
        pm_sample.write_log("sampler-sample");

        // STEP 1: Sample which strata to get next sample
        let index = super::sample_weights_table(&weights_table);
        if index.is_none() {
            // stratified storage is empty, wait for data loading
            debug!("Sampler sleeps waiting for data loading");
            sleep(Duration::from_millis(1000));
            continue;
        }
        let index = index.unwrap();
        // STEP 2: Access the queue for the sampled strata
        let existing_receiver = {
            let read_strata = strata.read().unwrap();
            read_strata.get_out_queue(index)
        };
        let receiver = existing_receiver.unwrap();
        /*
        TODO: validate this block is unnecessary
        let receiver = {
            if let Some(receiver) = existing_receiver {
                receiver
            } else {
                let mut strata = strata.write().unwrap();
                let (_, receiver) = strata.create(index);
                receiver
            }
        };
        */
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
        pm3_total.resume();
        while sampled_example.is_none() {
            pm1.resume();
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
            pm1.pause();
            if recv.is_none() {
                break;
            }
            let (example, (score, version)) = recv.unwrap();
            let (updated_score, model_size) = {
                pm2.resume();
                let trees = model.read().unwrap();
                pm2.pause();
                pm3.resume();
                let model_size = trees.len();
                pm3.pause();
                pm4.resume();
                let inc_score: f32 = {
                    trees[version..model_size].iter().map(|tree| {
                        tree.get_leaf_prediction(&example)
                    }).sum()
                };
                pm4.pause();
                (score + inc_score, model_size)
            };
            *grid += get_weight(&example, updated_score);
            if *grid >= grid_size {
                sampled_example = Some((example.clone(), (updated_score, model_size)));
            }
            stats_update_s.send((index, (-1, -get_weight(&example, score) as f64)));
            updated_examples.send((example, (updated_score, model_size)));
            num_updated += 1;
            pm_update.update(1);
        }
        pm3_total.pause();
        // STEP 4: Send the sampled example to the buffer loader
        if sampled_example.is_none() {
            debug!("Sampling the stratum {} failed because it has too few examples", index);
            continue;
        }
        let sampled_example = sampled_example.unwrap();
        let sample_count = (*grid / grid_size) as u32;
        if sample_count > 0 {
            sampled_examples.send((sampled_example, sample_count));
            *grid -= grid_size * (sample_count as f32);
            num_sampled += sample_count;
            pm_sample.update(sample_count as usize);
        }

        if pm_total.get_duration() > 5.0 {
            let total = pm_total.get_duration();
            let s3_total = pm3_total.get_duration();
            let step1 = pm1.get_duration() / s3_total;
            let step2 = pm2.get_duration() / s3_total;
            let step3 = pm3.get_duration() / s3_total;
            let step4 = pm4.get_duration() / s3_total;
            let others = 1.0 - step1 - step2 - step3 - step4;
            debug!("sample-perf-breakdown, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}",
                   (num_updated as f32) / total, (num_sampled as f32) / total, s3_total / total,
                   step1, step2, step3, step4, others);
            pm_total.reset();
            pm3_total.reset();
            pm1.reset();
            pm2.reset();
            pm3.reset();
            pm4.reset();
            num_updated = 0;
            num_sampled = 0;
            pm_total.start();
        }

        /*
        TODO: set a flag to enable this block
        {
            let signal = sampling_signal.read().unwrap();
            if *signal == Signal::STOP {
                debug!("sampler exits");
                break;
            }
        }
        */
    }
}
