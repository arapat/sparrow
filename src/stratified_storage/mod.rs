mod strata;
mod assigners;
mod samplers;
mod gatherer;
pub mod serial_storage;

use std::cmp::max;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::thread::spawn;
use std::thread::sleep;
use std::time::Duration;
use rand;
use bincode::serialize;
use commons::bins::Bins;
use commons::channel;
use commons::channel::Receiver;
use commons::channel::Sender;
use commons::get_sign;
use commons::io::write_all;
use commons::ExampleWithScore;
use commons::Model;
use labeled_data::LabeledData;
use super::Example;
use super::TFeature;

use buffer_loader::LockedBuffer;
use buffer_loader::SampleMode;
use self::assigners::Assigners;
use self::samplers::Samplers;
use self::gatherer::Gatherer;
use self::serial_storage::SerialStorage;
use self::strata::Strata;


pub const SPEED_TEST: bool = false;

#[derive(Serialize, Deserialize, Debug)]
pub struct F64 {
    pub val: f64
}

impl PartialEq for F64 {
    fn eq(&self, other: &F64) -> bool {
        get_sign(self.val - other.val) == 0
    }
}
impl Clone for F64 {
    fn clone(&self) -> F64 {
        F64 {
            val: self.val,
        }
    }
}
impl Eq for F64 {}

pub type WeightTableRead = evmap::ReadHandle<i8, Box<F64>>;
pub type WeightTableWrite = evmap::WriteHandle<i8, Box<F64>>;
pub type CountTableRead = evmap::ReadHandle<i8, i32>;
pub type CountTableWrite = evmap::WriteHandle<i8, i32>;

pub struct StratifiedStorage {
    updated_examples_s: Sender<ExampleWithScore>,
    positive: String,
}


impl StratifiedStorage {
    /// Create the stratified storage structure.
    ///
    /// * `num_examples`: the total number of examples in the training data set
    /// * `feature_size`: the number of features of the training examples
    /// * `num_examples_per_block`: the number of examples to write back to disk in batch (explained below)
    /// * `disk_buffer_filename`: the name of the binary file for saving the examples in strata on disk
    /// If such file does not exist, it will be created
    /// * `num_assigners`: the number of threads that run the `Assigner`s (explained below)
    /// * `num_samplers`: the number of threads that run the `Sampler`s (explained below)
    /// * `sampled_examples`: the channle that the stratified storage sends the sampled examples to
    /// the buffer loader
    /// * `models`: the channel that the booster sends the latest models in
    ///
    /// Stratified storage organizes training examples according to their weights
    /// given current learning model.
    /// The examples are assigned to different strata so that the weight ratio of the examples
    /// within the same stratum does not exceed 2.
    /// Most examples in a stratum are stored on disk, while a small number of examples remains
    /// in memory to be writen to disk or just read from disk and ready to send out to the sampler.
    ///
    /// The overall structure of the stratified storage is as follow:
    ///
    /// ![](https://www.lucidchart.com/publicSegments/view/c87b7a50-5442-4a41-a601-3dfb49b16511/image.png)
    ///
    /// The `Assigner`s read examples with updated scores from the `Sampler` and write them back to
    /// the corresponding strata based on their new weights. The examples would be put into the
    /// `In Queue`s first till a proper number of examples are accumulated that belongs to the
    /// same strata, at that point they would be written into disk in batch.
    ///
    /// Meanwhile, a certain number of examples from each stratum are loaded into the memory
    /// from the disk and kept in `Out Queue`s.
    /// The `Sampler`s iteratively select a stratum with a probability that proportional to
    /// the sum of weights of all examples in that stratum, send its next sampled example to the memory
    /// buffer, and remove that example from strata.
    ///
    /// A `Shared Weight Table` maintains the sum of the weights of all examples in each stratum.
    /// The `Assigner`s increase the value in the `Shared Weight Table` when a new example is inserted into
    /// a stratum.
    /// The `Sampler`s use the weights in the `Shared Weight Table` to decide which stratum to read next and
    /// send its next sampled example to the memory buffer. After an example is processed, the `Sampler` also
    /// updates its weight, sends it to right stratum, and updates `Shared Weight Table` accordingly.
    pub fn new(
        init_model: Model,
        num_examples: usize,
        sample_capacity: usize,
        feature_size: usize,
        positive: String,
        num_examples_per_block: usize,
        disk_buffer_filename: &str,
        memory_buffer: LockedBuffer,
        current_sample_version: Arc<RwLock<usize>>,
        sample_mode: SampleMode,
        num_assigners: usize,
        num_samplers: usize,
        models: Receiver<Model>,
        channel_size: usize,
        sampler_state: Arc<RwLock<bool>>,
        debug_mode: bool,
        resume_training: bool,
        exp_name: String,
    ) -> StratifiedStorage {
        let snapshot_filename = "stratified.serde".to_string();
        if resume_training {
            debug!("prepare to resume an earlier stratified structure");
        }
        // Weights table
        let (counts_table_r, counts_table_w): (CountTableRead, CountTableWrite) = evmap::new();
        let (weights_table_r, weights_table_w): (WeightTableRead, WeightTableWrite) = evmap::new();
        // let (sampled_examples_s, sampled_examples_r): (Sender<((ExampleWithScore, u32), u32)>, _) =
        let (sampled_examples_s, sampled_examples_r) =
            channel::bounded(channel_size, "gather-samples");
        /* TODO: Set a flag to enable resuming weight tables and the latest sample  */
        /* The backup mechanism does not back up the slot indices for each stratum  */
        /* yet. Implement it before turning on this flag.                           */
        /* For now, all examples will be reset their scores to 0 during the         */
        /* initilization.                                                           */
        /*
        let ser_strata = {
            if resume_training {
                // read snapshot
                debug!("reading the snapshot");
                let (ser_strata, _tables): (Vec<u8>, (HashMap<i8, i32>, HashMap<i8, f64>)) =
                    deserialize(&read_all(&snapshot_filename)).unwrap();
                debug!("reading the tables");
                let (counts_table, weights_table): (HashMap<_, _>, HashMap<_, f64>) = tables;
                // initialize the weight tables from the snapshot
                debug!("resume counts_table, {:?}", counts_table);
                debug!("resume weights_table, {:?}", weights_table);
                for (key, val) in counts_table.iter() {
                    counts_table_w.update(*key, *val);
                }
                for (key, val) in weights_table.iter() {
                    weights_table_w.update(*key, Box::new(F64 { val: *val }));
                }
                {
                    counts_table_w.refresh();
                    weights_table_w.refresh();
                }
                debug!("refreshed table");
                // Send out the last sample before creating the snapshot to the scanners
                {
                    let (version, last_sample): (usize, Vec<ExampleWithScore>) =
                        deserialize(&read_all(&"latest_sample.bin".to_string())).unwrap();
                    debug!("Loaded examples from the Sample version {}. \
                            Will send out to the scanners", version);
                    let sampled_examples = sampled_examples.clone();
                    spawn(move || {
                        last_sample.into_iter().for_each(|t| sampled_examples.send(((t, 1), 1)));
                    });
                }
                // serialized strata will be used for initialization in its `new` func
                Some(ser_strata)
            } else {
                None
            }
        };
        */
        let ser_strata = None;
        // Maintains weight tables
        let stats_update_s = start_update_weights_table(
            counts_table_r.clone(), counts_table_w, weights_table_r.clone(), weights_table_w,
            debug_mode);
        // Maintains all example on disk and in memory
        let strata = Strata::new(
            num_examples, feature_size, num_examples_per_block, disk_buffer_filename,
            sampler_state.clone(), ser_strata, stats_update_s.clone());
        let strata = Arc::new(RwLock::new(strata));

        // Start assigners, samplers, and gatherers
        let model = Arc::new(RwLock::new(init_model));
        {
            let model = model.clone();
            spawn(move || {
                while let Some(new_model) = models.recv() {
                    let model_len = new_model.size();
                    {
                        let mut model = model.write().unwrap();
                        *model = new_model;
                    }
                    debug!("stratified, model updated, {}", model_len);
                }
            });
        }
        let gatherer = Gatherer::new(
            sampled_examples_r,
            sample_capacity,
            memory_buffer,
            current_sample_version,
            model.clone(),
            exp_name,
        );
        let assigners = Assigners::new(
            strata.clone(),
            stats_update_s.clone(),
            num_assigners,
            channel_size,
        );
        let samplers = Samplers::new(
            model,
            strata.clone(),
            sampled_examples_s,
            assigners.updated_examples_s.clone(),
            stats_update_s.clone(),
            weights_table_r.clone(),
            num_samplers,
            sampler_state.clone(),
        );
        gatherer.run(sample_mode);
        assigners.run();
        samplers.run();

        run_snapshot_thread(snapshot_filename, strata, counts_table_r, weights_table_r, sampler_state);

        StratifiedStorage {
            updated_examples_s: assigners.updated_examples_s.clone(),
            positive: positive,
            // core objects (that are not saved as the fields):
            //     strata, counts_table_r, weights_table_r
        }
    }

    pub fn init_stratified_from_file(
        &self,
        filename: String,
        size: usize,
        batch_size: usize,
        feature_size: usize,
        bins: Vec<Bins>,
        model: Model,
    ) {
        let mut reader = SerialStorage::new(
            filename.clone(),
            size,
            feature_size,
            true,
            self.positive.clone(),
            None,
        );
        let updated_examples_s = self.updated_examples_s.clone();
        spawn(move || {
            let mut index = 0;
            let mut last_report_length = 0;
            while index < size {
                reader.read_raw(batch_size).into_iter().for_each(|data| {
                    let features: Vec<TFeature> =
                        data.feature.iter().enumerate()
                            .map(|(idx, val)| {
                                bins[idx].get_split_index(*val)
                            }).collect();
                    let mapped_data = LabeledData::new(features, data.label);
                    let (score, (model_size, _)) = model.get_prediction(&mapped_data, 0);
                    updated_examples_s.send((mapped_data, (score, model_size)));
                });
                index += batch_size;
                if index - last_report_length > size / 10 {
                    debug!("init-stratified, progress, {}", index);
                    last_report_length = index;
                }
            }
            debug!("Raw data on disk has been loaded into the stratified storage, \
                    filename {}, capacity {}, feature size {}", filename, size, feature_size);
        });
    }
}


fn start_update_weights_table(
    counts_table_r: CountTableRead, counts_table_w: CountTableWrite,
    weights_table_r: WeightTableRead, weights_table_w: WeightTableWrite,
    debug_mode: bool) -> Sender<(i8, (i32, f64))> {
    let (stats_update_s, stats_update_r) = channel::bounded(5000000, "stats");
    // Updating
    {
        let counts_table_r = counts_table_r.clone();
        let weights_table_r = weights_table_r.clone();
        let mut counts_table_w = counts_table_w;
        let mut weights_table_w = weights_table_w;
        spawn(move || {
            while let Some((index, (count, weight))) = stats_update_r.recv() {
                let val = counts_table_r.get_and(&index, |vs| vs[0]).unwrap_or(0);
                counts_table_w.update(index, val + count);
                let cur = weights_table_r.get_and(&index, |vs: &[Box<F64>]| vs[0].val)
                                            .unwrap_or(0.0);
                weights_table_w.update(index, Box::new(F64 { val: cur + weight }));
                {
                    counts_table_w.refresh();
                    weights_table_w.refresh();
                }
            }
        });
    }

    // Monitor the distribution of strata
    if debug_mode {
        let counts_table_r = counts_table_r.clone();
        let weights_table_r = weights_table_r.clone();
        spawn(move || {
            loop {
                sleep(Duration::from_millis(5000));
                let mut p: Vec<(i8, f64)> =
                    weights_table_r.map_into(|a: &i8, b: &[Box<F64>]| (a.clone(), b[0].val));
                p.sort_by(|a, b| (a.0).cmp(&b.0));
                let mut c: Vec<(i8, i32)> = counts_table_r.map_into(|a, b| (a.clone(), b[0]));
                c.sort_by(|a, b| (a.0).cmp(&b.0));
                let mut sump: f64 = p.iter().map(|t| t.1).sum();
                if get_sign(sump) == 0 {
                    sump = 1.0;
                }
                let ps: Vec<String> = p.into_iter()
                                        .map(|(idx, w)| (idx, 100.0 * w / sump))
                                        .map(|(idx, w)| format!("({}, {:.2})", idx, w))
                                        .collect();
                debug!("strata weights distr, {}, {}", ps.join(", "), sump);
                let sumc: i32 = max(c.iter().map(|t| t.1).sum(), 1);
                let cs: Vec<String> = c.into_iter()
                                        .map(|(idx, c)| (idx, 100.0 * c as f32 / (sumc as f32)))
                                        .map(|(idx, c)| format!("({}, {:.2})", idx, c))
                                        .collect();
                debug!("strata counts distr, {}, {}", cs.join(", "), sumc);
            }
        });
    }
    stats_update_s
}


fn run_snapshot_thread(
    filename: String,
    strata: Arc<RwLock<Strata>>,
    counts_table_r: CountTableRead,
    weights_table_r: WeightTableRead,
    sampler_state: Arc<RwLock<bool>>,
) {
    spawn(move || {
        let mut state = true;
        while state {
            state = {
                *(sampler_state.read().unwrap())
            };
            if !state {
                info!("Sampler has stopped. Prepare to take a snapshot of the stratified storage.");
                let ser_strata = {
                    let strata = strata.read().unwrap();
                    strata.serialize()
                };
                info!("Snapshot of the strata has been taken");
                let tables = {
                    let counts_table: HashMap<i8, i32> =
                        counts_table_r.map_into(|&k, vs| (k, vs[0]));
                    let weights_table: HashMap<i8, f64> =
                        weights_table_r.map_into(|&k, vs| (k, vs[0].val));
                    (counts_table, weights_table)
                };
                let data = serialize(&(ser_strata, tables)).unwrap();
                write_all(&filename, &data)
                    .expect("Failed to write the serialized stratified storage");
                info!("Snapshot of the stratified storage has been taken.");
            } else {
                sleep(Duration::from_secs(60));
            }
        }
    });
}


fn sample_weights_table(weights_table_r: &WeightTableRead) -> Option<i8> {
    let p: Vec<(i8, f64)> = weights_table_r.map_into(|a, b| (a.clone(), b[0].val));
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


#[cfg(test)]
mod tests {
    extern crate env_logger;

    use std::fs::remove_file;
    use commons::channel;

    use std::thread::spawn;
    use labeled_data::LabeledData;
    use commons::ExampleWithScore;
    use commons::Signal;
    use commons::performance_monitor::PerformanceMonitor;
    use super::StratifiedStorage;
    use ::TFeature;

    #[test]
    fn test_mean() {
        let _ = env_logger::try_init();
        let filename = "unittest-stratified3.bin";
        let batch = 100000;
        let num_read = 1000000;
        let (sampled_examples_send, sampled_examples_recv) = channel::bounded(1000, "sampled-examples");
        let (_, models_recv) = channel::bounded(10, "updated-models");
        let (signal_s, signal_r) = channel::bounded(10, "sampling-signal");
        signal_s.send(Signal::START);
        let stratified_storage = StratifiedStorage::new(
            batch * 10, 1, "1".to_string(), 10000, filename, 4, 4,
            sampled_examples_send, signal_r, models_recv, 10, false,
        );
        let updated_examples_send = stratified_storage.updated_examples_s.clone();
        let mut pm_load = PerformanceMonitor::new();
        pm_load.start();
        let loading = spawn(move || {
            for _ in 0..batch {
                for i in 1..11 {
                    let t = get_example(vec![i as TFeature], i as f32);
                    updated_examples_send.send(t.clone());
                }
            }
            println!("Loading speed: {}", (batch * 10) as f32 / pm_load.get_duration());
        });

        let mut pm_sample = PerformanceMonitor::new();
        pm_sample.start();
        let mut average = 0.0;
        for _ in 0..num_read {
            let recv = sampled_examples_recv.recv().unwrap().0;  // .1 is # of scanned examples
            average += (((recv.0).0).feature[0] as f32) * (recv.1 as f32) / (num_read as f32);
            pm_sample.update(recv.1 as usize);
        }
        spawn(move || {
            println!("Sampling speed: {}", num_read as f32 / pm_sample.get_duration());
        });
        let answer =
            (1..11).map(|a| a as f32).map(|a| a * a).sum::<f32>() / ((1..11).sum::<i32>() as f32);
        loading.join().unwrap();
        if (average - answer).abs() > 0.05 {
            spawn(move || {
                println!("Average: {}. Expect: {}.", average, answer);
            }).join().unwrap();
            assert!(false);
        }
        remove_file(filename).unwrap();
    }

    fn get_example(feature: Vec<TFeature>, weight: f32) -> ExampleWithScore {
        let label: i8 = 1;
        let example = LabeledData::new(feature, label);
        let score = -weight.ln();
        (example, (score, 0))
    }
}
