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