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


pub struct Stratified {
    assigners: Assigners,
    strata: Arc<RwLock<Strata>>,
    selectors: Selectors,
    weights_table: WeightsTable,
}


impl Stratified {
    pub fn new(num_examples: usize,
               feature_size: usize,
               num_examples_per_block: usize,
               updated_examples: Receiver<ExampleWithScore>,
               loaded_examples: Sender<ExampleWithScore>) -> Stratified {
        let strata = Arc::new(RwLock::new(
                Strata::new(num_examples, feature_size, num_examples_per_block, "stratified_buffer.bin")));
        let weights_table = Arc::new(RwLock::new(
                HashMap::new()));
        let assigners = Assigners::new(weights_table.clone(), updated_examples, strata.clone());
        let selectors = Selectors::new(weights_table.clone(), strata.clone(), loaded_examples);
        Stratified {
            assigners: assigners,
            strata: strata,
            selectors: selectors,
            weights_table: weights_table
        }
    }

    pub fn run(&mut self, num_assigners: usize, num_selectors: usize) {
        self.assigners.run(num_assigners);
        self.selectors.run(num_selectors);
    }
}