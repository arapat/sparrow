mod strata;
mod assigners;
mod samplers;
pub mod serial_storage;

use chan;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::mpsc::Sender;
use std::sync::mpsc::Receiver;

use commons::ExampleWithScore;
use commons::Model;
use super::Example;

use self::assigners::Assigners;
use self::samplers::Samplers;
use self::serial_storage::SerialStorage;
use self::strata::Strata;


type SharedCountsTable = Arc<RwLock<HashMap<i8, u32>>>;
type SharedWeightsTable = Arc<RwLock<HashMap<i8, f64>>>;


pub struct StratifiedStorage {
    // num_examples: usize,
    // feature_size: usize,
    // num_examples_per_block: usize,
    // disk_buffer_filename: String,
    // counts_table: SharedCountsTable,
    // weights_table: SharedWeightsTable,
    // num_assigners: usize,
    // num_samplers: usize,
    // assigners: Assigners,
    // samplers: Samplers,
    // updated_examples_r: chan::Receiver<ExampleWithScore>,
    updated_examples_s: chan::Sender<ExampleWithScore>,
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
        sampled_examples: Sender<ExampleWithScore>,
        models: Receiver<Model>,
    ) -> StratifiedStorage {
        let strata = Strata::new(num_examples, feature_size, num_examples_per_block,
                                 disk_buffer_filename);
        let strata = Arc::new(RwLock::new(strata));

        let counts_table = Arc::new(RwLock::new(HashMap::new()));
        let weights_table = Arc::new(RwLock::new(HashMap::new()));

        let (updated_examples_s, updated_examples_r) = chan::async();

        let mut assigners = Assigners::new(
            counts_table.clone(), weights_table.clone(), updated_examples_r.clone(),
            strata.clone());
        let mut samplers = Samplers::new(
            counts_table.clone(), weights_table.clone(), strata.clone(),
            sampled_examples, updated_examples_s.clone(), models);
        assigners.run(num_assigners);
        samplers.run(num_samplers);

        StratifiedStorage {
            // num_examples: num_examples,
            // feature_size: feature_size,
            // num_examples_per_block: num_examples_per_block,
            // disk_buffer_filename: String::from(disk_buffer_filename),
            // counts_table: counts_table,
            // weights_table: weights_table,
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
            filename,
            size,
            feature_size,
            is_binary,
            bytes_per_example,
            true,
        );

        let mut index = 0;
        while index < size {
            reader.read(batch_size).into_iter().for_each(|data| {
                self.updated_examples_s.send((data, (0.0, 0)));
            });
            index += batch_size;
        }
    }
}


#[cfg(test)]
mod tests {
    use std::fs::remove_file;
    use std::sync::mpsc;
    use std::collections::HashMap;

    use labeled_data::LabeledData;
    use commons::ExampleWithScore;
    use super::StratifiedStorage;

    #[test]
    fn test_stratified_one_by_one_1_thread() {
        let filename = "unittest-stratified1.bin";
        let (sampled_examples_send, sampled_examples_recv) = mpsc::channel();
        let (_, models_recv) = mpsc::channel();
        let stratified_storage = StratifiedStorage::new(
            1000, 3, 10, filename, 1, 1, sampled_examples_send, models_recv
        );
        let updated_examples_send = stratified_storage.updated_examples_s.clone();
        for i in 0..10 {
            let t = get_example(vec![0, 0, i], 1.0);
            updated_examples_send.send(t.clone());
        }
        let mut freq = HashMap::new();
        for _ in 0..100 {
            let recv = sampled_examples_recv.recv().unwrap();
            let c = freq.entry((recv.0).feature[2]).or_insert(0u8);
            *c += 1;
        }
        for (_, v) in freq.iter() {
            assert_eq!(*v, 20);
        }
        remove_file(filename).unwrap();
    }

    #[test]
    fn test_stratified_one_by_one_10_threads() {
        let filename = "unittest-stratified2.bin";
        let (sampled_examples_send, sampled_examples_recv) = mpsc::channel();
        let (_, models_recv) = mpsc::channel();
        let stratified_storage = StratifiedStorage::new(
            1000, 3, 10, filename, 1, 1, sampled_examples_send, models_recv
        );
        let updated_examples_send = stratified_storage.updated_examples_s.clone();
        for i in 0..10 {
            let t = get_example(vec![0, 0, i], 1.0);
            updated_examples_send.send(t.clone());
        }
        let mut freq = HashMap::new();
        for _ in 0..100 {
            let recv = sampled_examples_recv.recv().unwrap();
            let c = freq.entry((recv.0).feature[2]).or_insert(0u8);
            *c += 1;
        }
        for (_, v) in freq.iter() {
            assert_eq!(*v, 20);
        }
        remove_file(filename).unwrap();
    }

    fn get_example(feature: Vec<u8>, weight: f32) -> ExampleWithScore {
        let label: u8 = 0;
        let example = LabeledData::new(feature, label);
        let score = weight.ln();
        (example, (score, 0))
    }
}
