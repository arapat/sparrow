mod strata;
mod assigners;
mod selectors;
// mod mpmc_map;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use chan::Sender;
use chan::Receiver;

use commons::ExampleWithScore;
use self::assigners::Assigners;
use self::selectors::Selectors;
use self::strata::Strata;


type WeightsTable = Arc<RwLock<HashMap<i8, f64>>>;
type CountsTable = Arc<RwLock<HashMap<i8, u32>>>;


/// Start the stratified storage structure.
///
/// * `num_examples`: the total number of examples in the training data set
/// * `feature_size`: the number of features of the training examples
/// * `num_examples_per_block`: the number of examples to write back to disk in batch (explained below)
/// * `disk_buffer_filename`: the name of the binary file for saving the examples in strata on disk
/// If such file does not exist, it will be created
/// * `num_assigners`: the number of threads that run the `Assigner`s (explained below)
/// * `num_selectors`: the number of threads that run the `Selector`s (explained below)
/// * `updated_examples`: the channel that the sampler sends the examples with updated scores
/// * `loaded_examples`: the channle that the stratified storage sends the examples to be read by the `Sampler`.
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
/// The `Selector`s iteratively select a stratum with a probability that proportional to
/// the sum of weights of all examples in that stratum, send its next loaded example to the `Sampler`,
/// and remove that example from strata.
///
/// A `Shared Weight Table` maintains the sum of the weights of all examples in each stratum.
/// The `Assigner`s increase the value in the `Shared Weight Table` when a new example is inserted into
/// a stratum.
/// The `Selector`s use the weights in the `Shared Weight Table` to decide which stratum to read next and
/// send its next loaded example to the `Sampler`. After an example is selected, the `Selector` then
/// substracts the weight of that example from the corresponding stratum in the `Shared Weight Table`.
pub fn run_stratified(
        num_examples: usize,
        feature_size: usize,
        num_examples_per_block: usize,
        disk_buffer_filename: &str,
        num_assigners: usize,
        num_selectors: usize,
        updated_examples: Receiver<ExampleWithScore>,
        loaded_examples: Sender<ExampleWithScore>) -> (CountsTable, WeightsTable) {
    let strata = Arc::new(RwLock::new(
            Strata::new(num_examples, feature_size, num_examples_per_block, disk_buffer_filename)));
    let counts_table = Arc::new(RwLock::new(HashMap::new()));
    let weights_table = Arc::new(RwLock::new(HashMap::new()));
    let mut assigners = Assigners::new(
        counts_table.clone(), weights_table.clone(), updated_examples, strata.clone());
    let mut selectors = Selectors::new(
        counts_table.clone(), weights_table.clone(), strata.clone(), loaded_examples);
    assigners.run(num_assigners);
    selectors.run(num_selectors);
    (counts_table, weights_table)
}


#[cfg(test)]
mod tests {
    use std::fs::remove_file;
    use chan;

    use labeled_data::LabeledData;
    use commons::ExampleWithScore;
    use super::run_stratified;

    #[test]
    fn test_stratified_one_by_one_1_thread() {
        let filename = "unittest-stratified1.bin";
        let (updated_examples_send, updated_examples_recv) = chan::sync(10);
        let (loaded_examples_send, loaded_examples_recv) = chan::sync(10);
        let (counts_table, _) = run_stratified(
            1000, 3, 10, filename, 1, 1, updated_examples_recv, loaded_examples_send);
        for i in 0..2 {
            for k in 0..5 {
                let t = get_example(vec![0, i, k], (2.0f32).powi(k as i32));
                updated_examples_send.send(t.clone());
                let retrieve = loaded_examples_recv.recv().unwrap();
                assert_eq!(t, retrieve);
            }
        }
        let num_examples: u32 = counts_table.read().unwrap().values().sum();
        assert_eq!(num_examples, 0);
        remove_file(filename).unwrap();
    }

    #[test]
    fn test_stratified_one_by_one_10_threads() {
        let filename = "unittest-stratified2.bin";
        let (updated_examples_send, updated_examples_recv) = chan::sync(10);
        let (loaded_examples_send, loaded_examples_recv) = chan::sync(10);
        let (counts_table, _) = run_stratified(
            1000, 3, 10, filename, 10, 10, updated_examples_recv, loaded_examples_send);
        for i in 0..2 {
            for k in 0..5 {
                let t = get_example(vec![0, i, k], (2.0f32).powi(k as i32));
                updated_examples_send.send(t.clone());
                let retrieve = loaded_examples_recv.recv().unwrap();
                assert_eq!(t, retrieve);
            }
        }
        let num_examples: u32 = counts_table.read().unwrap().values().sum();
        assert_eq!(num_examples, 0);
        remove_file(filename).unwrap();
    }

    #[test]
    fn test_stratified_seq_1_thread() {
        let filename = "unittest-stratified3.bin";
        let (updated_examples_send, updated_examples_recv) = chan::sync(10);
        let (loaded_examples_send, loaded_examples_recv) = chan::sync(10);
        let (counts_table, _) = run_stratified(
            1000, 3, 5, filename, 1, 1, updated_examples_recv, loaded_examples_send);
        let mut solution = vec![];
        for i in 0..5 {
            for k in 0..5 {
                let t = get_example(vec![0, i, k], (2.0f32).powi(k as i32));
                updated_examples_send.send(t.clone());
                solution.push(t);
            }
        }

        let mut answer = vec![];
        for _ in 0..5 {
            for _ in 0..5 {
                let retrieve = loaded_examples_recv.recv().unwrap();
                answer.push(retrieve);
            }
        }

        solution.sort_by_key(|t| (t.0).get_features().clone());
        answer.sort_by_key(|t| (t.0).get_features().clone());
        for (a, b) in solution.iter().zip(answer.iter()) {
            assert_eq!(a, b);
        }

        let num_examples: u32 = counts_table.read().unwrap().values().sum();
        assert_eq!(num_examples, 0);
        remove_file(filename).unwrap();
    }

    #[test]
    fn test_stratified_seq_10_threads() {
        let filename = "unittest-stratified4.bin";
        let (updated_examples_send, updated_examples_recv) = chan::sync(10);
        let (loaded_examples_send, loaded_examples_recv) = chan::sync(10);
        let (counts_table, _) = run_stratified(
            3000, 3, 5, filename, 10, 10, updated_examples_recv, loaded_examples_send);
        let mut solution = vec![];
        for i in 0..50 {
            for k in 0..50 {
                let t = get_example(vec![0, i, k], (2.0f32).powi(k as i32));
                updated_examples_send.send(t.clone());
                solution.push(t);
            }
        }

        let mut answer = vec![];
        for _ in 0..50 {
            for _ in 0..50 {
                let retrieve = loaded_examples_recv.recv().unwrap();
                answer.push(retrieve);
            }
        }

        solution.sort_by_key(|t| (t.0).get_features().clone());
        answer.sort_by_key(|t| (t.0).get_features().clone());
        for (a, b) in solution.iter().zip(answer.iter()) {
            assert_eq!(a, b);
        }

        let num_examples: u32 = counts_table.read().unwrap().values().sum();
        assert_eq!(num_examples, 0);
        remove_file(filename).unwrap();
    }

    fn get_example(features: Vec<u8>, weight: f32) -> ExampleWithScore {
        let label: u8 = 0;
        let example = LabeledData::new(features, label);
        let score = weight.ln();
        (example, (score, 0))
    }
}