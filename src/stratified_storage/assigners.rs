
use std::sync::Arc;
use std::sync::RwLock;
use chan::Receiver;

use std::thread::spawn;

use commons::ExampleWithScore;
use commons::performance_monitor::PerformanceMonitor;
use super::Strata;
use super::SharedCountsTable;
use super::SharedWeightsTable;

use commons::get_weight;


pub struct Assigners {
    counts_table: SharedCountsTable,
    weights_table: SharedWeightsTable,
    updated_examples_out: Receiver<ExampleWithScore>,
    strata: Arc<RwLock<Strata>>
}


impl Assigners {
    pub fn new(
        counts_table: SharedCountsTable,
        weights_table: SharedWeightsTable,
        updated_examples_out: Receiver<ExampleWithScore>,
        strata: Arc<RwLock<Strata>>,
    ) -> Assigners {
        Assigners {
            counts_table: counts_table,
            weights_table: weights_table,
            updated_examples_out: updated_examples_out,
            strata: strata
        }
    }

    pub fn run(&mut self, num_threads: usize) {
        for _ in 0..num_threads {
            let counts_table = self.counts_table.clone();
            let weights_table = self.weights_table.clone(); 
            let updated_examples_out = self.updated_examples_out.clone();
            let strata = self.strata.clone();
            spawn(move|| {
                let mut pm = PerformanceMonitor::new();
                pm.start();
                while let Some(ret) = updated_examples_out.recv() {
                    let (example, (score, version)) = ret;
                    let weight = get_weight(&example, score);
                    let index = weight.log2() as i8;
                    let read_strata = strata.read().unwrap();
                    let mut sender = read_strata.get_in_queue(index);
                    drop(read_strata);
                    if sender.is_none() {
                        let mut strata = strata.write().unwrap();
                        sender = Some(strata.create(index).0);
                        drop(strata);
                    }
                    sender.unwrap().send((example, (score, version))).unwrap();
                    {
                        let mut weights = weights_table.write().unwrap();
                        let prev_weight = weights.entry(index).or_insert(0.0);
                        *prev_weight += weight as f64;

                        let mut counts = counts_table.write().unwrap();
                        let prev_count = counts.entry(index).or_insert(0);
                        *prev_count += 1;
                    }
                    pm.update(1);
                    pm.write_log("selector");
                }
                error!("Updated examples queue was closed.");
            });
        }
    }
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::RwLock;
    use std::time::Duration;
    use chan::Sender;

    use std::fs::remove_file;
    use std::thread::sleep;
    use chan;

    use labeled_data::LabeledData;
    use commons::ExampleWithScore;
    use super::super::Strata;
    use super::Assigners;
    use super::SharedCountsTable;

    #[test]
    fn test_assigner_1_thread() {
        let filename = "unittest-assigners1.bin";
        let (counts_table, sender, mut assigners) = get_assigner(filename);
        assigners.run(1);
        for i in 0..1 {
            for k in 0..3 {
                let t = get_example(vec![0, i, k], (2.0f32).powi(k as i32));
                sender.send(t.clone());
            }
        }

        sleep(Duration::from_millis(500));
        let num_examples: u32 = counts_table.read().unwrap().values().sum();
        assert_eq!(num_examples, 3);
        remove_file(filename).unwrap();
    }

    #[test]
    fn test_assigner_10_thread() {
        let filename = "unittest-assigners10.bin";
        let (counts_table, sender, mut assigners) = get_assigner(filename);
        assigners.run(10);
        for i in 0..10 {
            for k in 0..3 {
                let t = get_example(vec![0, i, k], (2.0f32).powi(k as i32));
                sender.send(t.clone());
            }
        }

        sleep(Duration::from_millis(500));
        let num_examples: u32 = counts_table.read().unwrap().values().sum();
        assert_eq!(num_examples, 30);
        remove_file(filename).unwrap();
    }

    fn get_assigner(filename: &str) -> (SharedCountsTable, Sender<ExampleWithScore>, Assigners) {
        let strata = Arc::new(RwLock::new(
                Strata::new(100, 3, 10, filename)));
        let counts_table = Arc::new(RwLock::new(HashMap::new()));
        let weights_table = Arc::new(RwLock::new(HashMap::new()));
        let (updated_examples_send, updated_examples_recv) = chan::sync(10);
        (counts_table.clone(),
         updated_examples_send,
         Assigners::new(counts_table, weights_table, updated_examples_recv, strata))
    }

    fn get_example(features: Vec<u8>, weight: f32) -> ExampleWithScore {
        let label: u8 = 0;
        let example = LabeledData::new(features, label);
        let score = weight.ln();
        (example, (score, 0))
    }
}