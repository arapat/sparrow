mod gatherer;

use rayon::prelude::*;
use crossbeam_channel as channel;

use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;
use self::channel::Receiver;

use std::cmp::min;
use std::thread::sleep;

use commons::ExampleInSampleSet;
use commons::ExampleWithScore;
use commons::Model;
// use commons::performance_monitor::PerformanceMonitor;

use commons::get_weight;
use self::gatherer::run_gatherer;


/// Double-buffered sample set. It consists of two buffers stores in memory. One of the
/// buffer is used for providing examples to the boosting algorithm, meanwhile the other
/// would receive new examples from the sampler module. Once the other buffer is filled,
/// the functions of the two buffers switch so that the sample set used for training
/// would be updated periodically.
///
/// It might take several scans over the sample set before the new sample set is ready.
/// Thus it is very likely that the examples in the sample set would be accessed multiple times.
#[derive(Debug)]
pub struct BufferLoader {
    size: usize,
    batch_size: usize,
    num_batch: usize,

    examples: Vec<ExampleInSampleSet>,
    pub new_examples: Arc<RwLock<Option<Vec<ExampleWithScore>>>>,

    pub ess: f32,

    curr_example: usize,
    // performance: PerformanceMonitor
}


impl BufferLoader {
    /// Create a new BufferLoader.
    ///
    /// `size`: the size of sample set. The total size of the two buffers would be 2x of `size`.
    /// `batch_size`: the number of examples that feeds to the boosting algorithm at a time.
    ///
    /// If `init_block` is set to `true`, this function would block until the first sample
    /// set is created (i.e. enough examples is received from the sampler).
    /// Alternatively, one can explicitly call the `load` function to create the first sample set
    /// before start training the boosting algorithm.
    pub fn new(
        size: usize,
        batch_size: usize,
        gather_new_sample: Option<Receiver<(ExampleWithScore, u32)>>,
        init_block: bool
    ) -> BufferLoader {
        let new_examples = Arc::new(RwLock::new(None));
        if gather_new_sample.is_some() {
            run_gatherer(gather_new_sample.unwrap(), new_examples.clone(), size);
        }
        let num_batch = (size + batch_size - 1) / batch_size;
        let mut buffer_loader = BufferLoader {
            size: size,
            batch_size: batch_size,
            num_batch: num_batch,

            examples: vec![],
            new_examples: new_examples,

            ess: 1.0,

            curr_example: 0,
            // performance: PerformanceMonitor::new()
        };
        if init_block {
            buffer_loader.load();
        }
        buffer_loader
    }

    /// Create the first sample set reading from the sampler.
    /// This function blocks until the first sample set is created.
    pub fn load(&mut self) {
        while !self.try_switch() {
            sleep(Duration::from_millis(1000));
        }
    }

    /// Return the number of batches (i.e. the number of function calls to `get_next_batch`)
    /// before exhausting the current sample set.
    #[allow(dead_code)]
    pub fn get_num_batches(&self) -> usize {
        self.num_batch
    }

    /// Read next batch of examples.
    ///
    /// `allow_switch`: If set to true, the alternate buffer would be checked if it
    /// was ready. If it was, the loader will switched to the alternate buffer for
    /// reading the next batch of examples.
    pub fn get_next_batch(&mut self, allow_switch: bool) -> &[ExampleInSampleSet] {
        if allow_switch {
            self.try_switch();
        }
        self.curr_example += self.batch_size;
        if self.curr_example >= self.size {
            self.update_ess();
            self.curr_example = 0;
        }

        assert!(!self.examples.is_empty());
        let tail = min(self.curr_example + self.batch_size, self.size);
        &self.examples[self.curr_example..tail]
    }

    /// Update the scores of the examples in the current batch using `model`.
    pub fn update_scores(&mut self, model: &Model) {
        let model_size = model.len();
        self.examples.par_iter_mut().for_each(|example| {
            let mut curr_score = (example.2).0;
            for tree in model[((example.2).1)..model_size].iter() {
                curr_score += tree.get_leaf_prediction(&example.0);
            }
            (*example).2 = (curr_score, model_size);
        });
        self.update_ess();
    }

    fn try_switch(&mut self) -> bool {
        let new_examples = {
            if let Ok(mut new_examples) = self.new_examples.try_write() {
                new_examples.take()
            } else {
                None
            }
        };
        if new_examples.is_some() {
            self.examples = new_examples.unwrap()
                                        .into_iter()
                                        .map(|t| {
                                            let (a, s) = t;
                                            (a, s, s.clone())
                                        }).collect();
            self.curr_example = 0;
            debug!("switched-buffer, {}", self.examples.len());
            true
        } else {
            false
        }
    }

    // ESS and others
    /// Get the estimate of the effective sample size of the current sample set.
    fn update_ess(&mut self) {
        let (sum_weights, sum_weight_squared) =
            self.examples.iter()
                        .map(|(data, (base_score, _), (curr_score, _))| {
                            let score = curr_score - base_score;
                            let w = get_weight(data, score);
                            (w, w * w)
                        })
                        .fold((0.0, 0.0), |acc, x| (acc.0 + x.0, acc.1 + x.1));
        self.ess = sum_weights.powi(2) / sum_weight_squared / (self.size as f32);
        trace!("loader-reset, {}", self.ess);
    }
}


#[cfg(test)]
mod tests {
    use crossbeam_channel as channel;
    use std::thread::sleep;

    use std::time::Duration;

    use labeled_data::LabeledData;
    use commons::ExampleWithScore;
    use super::BufferLoader;


    #[test]
    fn test_buffer_loader() {
        let (sender, receiver) = channel::bounded(10);
        let mut buffer_loader = BufferLoader::new(100, 10, Some(receiver), false);
        sender.send((get_example(vec![0, 1, 2], 1.0), 100));
        sleep(Duration::from_millis(1000));
        for _ in 0..20 {
            let batch = buffer_loader.get_next_batch(true);
            assert_eq!(batch.len(), 10);
            assert_eq!((batch[0].1).0, 1.0);
            assert_eq!((batch[0].2).0, 1.0);
            assert_eq!((batch[9].1).0, 1.0);
            assert_eq!((batch[9].2).0, 1.0);
        }
        sender.send((get_example(vec![0, 1, 2], 2.0), 100));
        sleep(Duration::from_millis(1000));
        for _ in 0..10 {
            let batch = buffer_loader.get_next_batch(true);
            assert_eq!(batch.len(), 10);
            assert_eq!((batch[0].1).0, 2.0);
            assert_eq!((batch[0].2).0, 2.0);
            assert_eq!((batch[9].1).0, 2.0);
            assert_eq!((batch[9].2).0, 2.0);
        }
    }

    fn get_example(features: Vec<u8>, score: f32) -> ExampleWithScore {
        let label: u8 = 0;
        let example = LabeledData::new(features, label);
        (example, (score, 0))
    }
}
