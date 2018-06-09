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
    use std::collections::HashMap;

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
        let (counts_table, weights_table) = run_stratified(
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
        let (counts_table, weights_table) = run_stratified(
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
        let (counts_table, weights_table) = run_stratified(
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
        for i in 0..5 {
            for k in 0..5 {
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
        let (counts_table, weights_table) = run_stratified(
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
        for i in 0..50 {
            for k in 0..50 {
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