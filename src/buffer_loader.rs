use rand::Rng;
use rand::thread_rng;
use rayon::prelude::*;

use std::ops::Range;
use std::sync::Arc;
use std::sync::RwLock;
use std::thread::spawn;
use std::time::Duration;
use std::sync::mpsc::Receiver;

use std::thread::sleep;

use super::ExampleWithScore;

use commons::ExampleInSampleSet;
use commons::Model;
use commons::performance_monitor::PerformanceMonitor;

use commons::get_weight;


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

    examples_in_use: Vec<ExampleInSampleSet>,
    examples_in_cons: Arc<RwLock<Vec<ExampleInSampleSet>>>,
    is_cons_ready: Arc<RwLock<bool>>,

    ess: Option<f32>,
    sum_weights: f32,
    sum_weight_squared: f32,

    _cursor: usize,
    _curr_range: Range<usize>,
    _scores_synced: bool,
    performance: PerformanceMonitor
}


impl BufferLoader {
    /// Create a new BufferLoader.
    ///
    /// `size`: the size of sample set. The total size of the two buffers would be 2x of `size`.
    /// `batch_size`: the number of examples that feeds to the boosting algorithm at a time.
    /// `sampled_examples`: the channel for receiving the newly sampled examples from the sampler.
    ///
    /// If `wait_till_loaded` is set to `true`, this function would block until the first sample
    /// set is created (i.e. enough examples is received from the sampler).
    /// Alternatively, one can explicitly call the `load` function to create the first sample set
    /// before start training the boosting algorithm.
    pub fn new(size: usize, batch_size: usize,
               sampled_examples: Receiver<ExampleWithScore>,
               wait_till_loaded: bool) -> BufferLoader {
        let examples_in_cons = Arc::new(RwLock::new(Vec::with_capacity(size)));
        let is_cons_ready = Arc::new(RwLock::new(false));
        {
            let capacity = size.clone();
            let cons_vec = examples_in_cons.clone();
            let cons_mark = is_cons_ready.clone();
            spawn(move|| {
                load_buffer(capacity, sampled_examples, cons_vec, cons_mark);
            });
        }

        let num_batch = (size + batch_size - 1) / batch_size;
        let mut buffer_loader = BufferLoader {
            size: size,
            batch_size: batch_size,
            num_batch: num_batch,

            examples_in_use: vec![],
            examples_in_cons: examples_in_cons,
            is_cons_ready: is_cons_ready,

            ess: None,
            sum_weights: 0.0,
            sum_weight_squared: 0.0,

            _cursor: 0,
            _curr_range: (0..0),
            _scores_synced: false,
            performance: PerformanceMonitor::new()
        };

        if wait_till_loaded {
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

    /// Return the number of batches (i.e. the number of function calls to `fetch_next_batch`)
    /// before exhausting the current sample set.
    #[allow(dead_code)]
    pub fn get_num_batches(&self) -> usize {
        self.num_batch
    }

    /// Get current batch of examples.
    pub fn get_curr_batch(&self, is_scores_updated: bool) -> &[ExampleInSampleSet] {
        assert!(!self.examples_in_use.is_empty());
        // scores must be updated unless is_scores_updated is not required.
        assert!(self._scores_synced || !is_scores_updated);
        let range = self._curr_range.clone();
        &self.examples_in_use[range]
    }

    /// Read next batch of examples.
    pub fn fetch_next_batch(&mut self, allow_switch: bool) {
        // self.performance.resume();

        if allow_switch {
            self.try_switch();
        }
        let (curr_loc, batch_size) = self.get_next_batch_size();
        self._curr_range = curr_loc..(curr_loc + batch_size);
        self._scores_synced = false;

        // self.performance.update(batch_size);
        // self.performance.pause();
    }

    /// Update the scores of the examples in the current batch using `model`.
    pub fn update_scores(&mut self, model: &Model) {
        assert!(!self.examples_in_use.is_empty());
        if self._scores_synced {
            return;
        }

        let model_size = model.len();
        let range = self._curr_range.clone();
        self.examples_in_use[range].par_iter_mut().for_each(|example| {
            let mut curr_score = (example.2).0;
            for tree in model[((example.2).1)..model_size].iter() {
                curr_score += tree.get_leaf_prediction(&example.0);
            }
            *example = (example.0.clone(), example.1, (curr_score, model_size));
        });
        self._scores_synced = true;
        self.update_stats_for_ess();
    }

    fn get_next_batch_size(&mut self) -> (usize, usize) {
        let curr_loc = self._cursor;
        let batch_size = if (self._cursor + 1) * self.batch_size < self.size {
            self._cursor += 1;
            self.batch_size
        } else {
            self.update_ess();
            let tail_remains = self.size - self._cursor * self.batch_size;
            self._cursor = 0;
            tail_remains
        };
        (curr_loc, batch_size)
    }

    fn try_switch(&mut self) -> bool {
        {
            let ready = self.is_cons_ready.try_read();
            if ready.is_err() || (*ready.unwrap()) == false {
                return false;
            }
        }
        {
            if let Ok(mut examples_in_cons) = self.examples_in_cons.write() {
                self.examples_in_use = examples_in_cons.to_vec();
                examples_in_cons.clear();
            }
            if let Ok(mut ready) = self.is_cons_ready.write() {
                *ready = false;
            }
        }
        self._cursor = 0;
        true
    }

    // ESS and others
    /// Get the estimate of the effective sample size of the current sample set.
    #[allow(dead_code)]
    pub fn get_ess(&self) -> Option<f32> {
        self.ess
    }

    fn update_stats_for_ess(&mut self) {
        let mut sum_weights        = 0.0;
        let mut sum_weight_squared = 0.0;
        self.get_curr_batch(true)
            .iter()
            .for_each(|(data, (base_score, _), (curr_score, _))| {
                let score = curr_score - base_score;
                let w = get_weight(data, score);
                sum_weights += w;
                sum_weight_squared += w * w;
            });
        self.sum_weights        += sum_weights;
        self.sum_weight_squared += sum_weight_squared;
    }

    fn update_ess(&mut self) {
        let count = self.size;
        let ess = self.sum_weights.powi(2) / self.sum_weight_squared / (count as f32);
        trace!("loader-reset, {}", ess);
        self.ess = Some(ess);
        self.sum_weights = 0.0;
        self.sum_weight_squared = 0.0;
    }

    /*
    fn report_timer(&mut self, timer: &mut PerformanceMonitor, timer_label: &str) {
        let (since_last_check, _, _, speed) = timer.get_performance();
        if since_last_check >= 300 {
            debug!("{}, {}, {}", timer_label, self.name, speed);
            timer.reset_last_check();
        }
    }
    */
}


fn load_buffer(capacity: usize,
               sampled_examples: Receiver<ExampleWithScore>,
               examples_in_cons: Arc<RwLock<Vec<ExampleInSampleSet>>>,
               is_cons_ready: Arc<RwLock<bool>>) {
    loop {
        // Make sure previous buffer has been received by the loader
        loop {
            let b = is_cons_ready.read().unwrap();
            if *b {
                drop(b);
                sleep(Duration::from_millis(1000));
            } else {
                break;
            }
        }
        // Fill the new buffer
        {
            if let Ok(mut examples) = examples_in_cons.write() {
                debug!("BUFFER LOADING START");
                let mut pm = PerformanceMonitor::new();
                pm.start();
                while examples.len() < capacity {
                    if let Ok((example, (score, node))) = sampled_examples.recv() {
                        examples.push((example, (score.clone(), node.clone()), (score, node)));
                    } else {
                        error!("Sampled examples queue is closed.");
                    }
                }
                thread_rng().shuffle(&mut *examples);
                pm.pause();
                debug!("BUFFER LOADING FINISH, {}", capacity as f32 / pm.get_duration());
            }
        }
        {
            let mut b = is_cons_ready.write().unwrap();
            *b = true;
        }
    }
}


#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread::sleep;
    use std::thread::spawn;

    use std::time::Duration;

    use labeled_data::LabeledData;
    use commons::ExampleWithScore;
    use super::BufferLoader;


    #[test]
    fn test_buffer_loader() {
        let (sender, receiver) = mpsc::sync_channel(100);

        let sender_clone = sender.clone();
        spawn(move|| {
            for _ in 0..100 {
                let t = get_example(vec![0, 1, 2], 1.0);
                sender_clone.send(t.clone()).unwrap();
            }
        });

        let mut buffer_loader = BufferLoader::new(100, 10, receiver, true);
        for _ in 0..100 {
            let t = get_example(vec![0, 1, 2], 2.0);
            sender.send(t.clone()).unwrap();
        }
        for _ in 0..10 {
            buffer_loader.fetch_next_batch(true);
            let batch = buffer_loader.get_curr_batch(false);
            assert_eq!(batch.len(), 10);
            assert_eq!((batch[0].1).0, 1.0);
            assert_eq!((batch[0].2).0, 1.0);
            assert_eq!((batch[9].1).0, 1.0);
            assert_eq!((batch[9].2).0, 1.0);
        }
        sleep(Duration::from_millis(1000));
        for _ in 0..10 {
            buffer_loader.fetch_next_batch(true);
            let batch = buffer_loader.get_curr_batch(false);
            assert_eq!(batch.len(), 10);
            assert_eq!((batch[0].1).0, 2.0);
            assert_eq!((batch[0].2).0, 2.0);
            assert_eq!((batch[9].1).0, 2.0);
            assert_eq!((batch[9].2).0, 2.0);
        }
    }

    #[test]
    #[should_panic]
    fn test_buffer_loader_should_panic() {
        let (sender, receiver) = mpsc::sync_channel(10);

        let sender_clone = sender.clone();
        spawn(move|| {
            for _ in 0..10 {
                let t = get_example(vec![0, 1, 2], 1.0);
                sender_clone.send(t.clone()).unwrap();
            }
        });

        let mut buffer_loader = BufferLoader::new(10, 3, receiver, true);
        buffer_loader.fetch_next_batch(true);
        buffer_loader.get_curr_batch(true);
    }

    fn get_example(features: Vec<u8>, score: f32) -> ExampleWithScore {
        let label: u8 = 0;
        let example = LabeledData::new(features, label);
        (example, (score, 0))
    }
}