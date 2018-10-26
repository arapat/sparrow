mod strata;
mod assigners;
mod samplers;
pub mod serial_storage;

use std::cmp::max;
use std::sync::Arc;
use std::sync::RwLock;
use std::thread::spawn;
use std::thread::sleep;
use std::time::Duration;
use evmap;
use rand;
use commons::channel;
use commons::get_sign;

use commons::ExampleWithScore;
use commons::Model;
use commons::channel::Receiver;
use commons::channel::Sender;
use super::Example;

use self::assigners::Assigners;
use self::samplers::Samplers;
use self::serial_storage::SerialStorage;
use self::strata::Strata;


pub const SPEED_TEST: bool = false;

pub struct F64 {
    pub val: f64
}

impl PartialEq for F64 {
    fn eq(&self, other: &F64) -> bool {
        get_sign(self.val - other.val) == 0
    }
}
impl Eq for F64 {}

type WeightTableRead = evmap::ReadHandle<i8, Box<F64>>;

pub struct StratifiedStorage {
    // num_examples: usize,
    // feature_size: usize,
    // num_examples_per_block: usize,
    // disk_buffer_filename: String,
    #[allow(dead_code)] counts_table_r: evmap::ReadHandle<i8, i32>,
    #[allow(dead_code)] weights_table_r: WeightTableRead,
    // num_assigners: usize,
    // num_samplers: usize,
    // assigners: Assigners,
    // samplers: Samplers,
    // updated_examples_r: channel::Receiver<ExampleWithScore>,
    updated_examples_s: Sender<ExampleWithScore>,
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
        num_examples: usize,
        feature_size: usize,
        num_examples_per_block: usize,
        disk_buffer_filename: &str,
        num_assigners: usize,
        num_samplers: usize,
        sampled_examples: Sender<(ExampleWithScore, u32)>,
        models: Receiver<Model>,
        channel_size: usize,
    ) -> StratifiedStorage {
        let strata = Strata::new(num_examples, feature_size, num_examples_per_block,
                                 disk_buffer_filename);
        let strata = Arc::new(RwLock::new(strata));

        let (counts_table_r, mut counts_table_w) = evmap::new();
        let (weights_table_r, mut weights_table_w) = evmap::new();
        let (updated_examples_s, updated_examples_r) = channel::bounded(channel_size, "updated-examples");
        // The messages in the stats channel are very small, so its capacity can be larger.
        let (stats_update_s, stats_update_r) = channel::bounded(5000000, "stats");
        {
            let counts_table_r = counts_table_r.clone();
            let weights_table_r = weights_table_r.clone();
            spawn(move || {
                while let Some((index, (count, weight))) = stats_update_r.recv() {
                    let val = counts_table_r.get_and(&index, |vs| vs[0]);
                    counts_table_w.update(index, val.unwrap_or(0) + count);
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
        {
            let counts_table_r = counts_table_r.clone();
            let weights_table_r = weights_table_r.clone();
            spawn(move || {
                loop {
                    sleep(Duration::from_millis(2000));
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

        let mut assigners = Assigners::new(
            updated_examples_r.clone(), strata.clone(), stats_update_s.clone());
        let mut samplers = Samplers::new(
            strata.clone(), sampled_examples, updated_examples_s.clone(), models,
            stats_update_s.clone(), weights_table_r.clone());
        assigners.run(num_assigners);
        samplers.run(num_samplers);

        StratifiedStorage {
            // num_examples: num_examples,
            // feature_size: feature_size,
            // num_examples_per_block: num_examples_per_block,
            // disk_buffer_filename: String::from(disk_buffer_filename),
            counts_table_r: counts_table_r,
            weights_table_r: weights_table_r,
            // num_assigners: num_assigners,
            // num_samplers: num_samplers,
            // assigners: assigners,
            // samplers: samplers,
            // updated_examples_r: updated_examples_r,
            updated_examples_s: updated_examples_s,
        }
    }

    pub fn init_stratified_from_file(
        self,
        filename: String,
        size: usize,
        batch_size: usize,
        feature_size: usize,
        is_binary: bool,
        bytes_per_example: Option<usize>,
    ) {
        let mut reader = SerialStorage::new(
            filename.clone(),
            size,
            feature_size,
            is_binary,
            bytes_per_example,
            true,
        );

        spawn(move || {
            let mut index = 0;
            while index < size {
                reader.read(batch_size).into_iter().for_each(|data| {
                    self.updated_examples_s.send((data, (0.0, 0)));
                });
                index += batch_size;
            }
            debug!("Raw data on disk has been loaded into the stratified storage,
                    filename {}, capacity {}, feature size {}", filename, size, feature_size);
        });
    }
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
    use commons::performance_monitor::PerformanceMonitor;
    use super::StratifiedStorage;

    #[test]
    fn test_mean() {
        let _ = env_logger::try_init();
        let filename = "unittest-stratified3.bin";
        let batch = 100000;
        let num_read = 1000000;
        let (sampled_examples_send, sampled_examples_recv) = channel::bounded(1000, "sampled-examples");
        let (_, models_recv) = channel::bounded(10, "updated-models");
        let stratified_storage = StratifiedStorage::new(
            batch * 10, 1, 10000, filename, 4, 4, sampled_examples_send, models_recv, 10
        );
        let updated_examples_send = stratified_storage.updated_examples_s.clone();
        let mut pm_load = PerformanceMonitor::new();
        pm_load.start();
        let loading = spawn(move || {
            for _ in 0..batch {
                for i in 1..11 {
                    let t = get_example(vec![i], i as f32);
                    updated_examples_send.send(t.clone());
                }
            }
            println!("Loading speed: {}", (batch * 10) as f32 / pm_load.get_duration());
        });

        let mut pm_sample = PerformanceMonitor::new();
        pm_sample.start();
        let mut average = 0.0;
        for _ in 0..num_read {
            let recv = sampled_examples_recv.recv().unwrap();
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

    fn get_example(feature: Vec<u8>, weight: f32) -> ExampleWithScore {
        let label: u8 = 1;
        let example = LabeledData::new(feature, label);
        let score = -weight.ln();
        (example, (score, 0))
    }
}
