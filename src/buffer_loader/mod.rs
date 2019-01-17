mod gatherer;

use rayon::prelude::*;

use std::cmp::min;
use std::sync::Arc;
use std::sync::RwLock;

use commons::channel::Receiver;
use commons::channel::Sender;
use commons::get_weight;
use commons::performance_monitor::PerformanceMonitor;
use commons::ExampleInSampleSet;
use commons::ExampleWithScore;
use commons::Model;
use commons::Signal;
use self::gatherer::Gatherer;


/// Double-buffered sample set. It consists of two buffers stores in memory. One of the
/// buffer is used for providing examples to the boosting algorithm, meanwhile the other
/// would receive new examples from the sampler module. Once the other buffer is filled,
/// the functions of the two buffers switch so that the sample set used for training
/// would be updated periodically.
///
/// It might take several scans over the sample set before the new sample set is ready.
/// Thus it is very likely that the examples in the sample set would be accessed multiple times.
pub struct BufferLoader {
    size: usize,
    batch_size: usize,
    num_batch: usize,

    examples: Vec<ExampleInSampleSet>,
    new_examples: Arc<RwLock<Option<Vec<ExampleWithScore>>>>,
    gatherer: Gatherer,
    serial_sampling: bool,
    sampling_signal_channel: Sender<Signal>,

    ess: f32,
    min_ess: f32,
    curr_example: usize,
    sampling_pm: PerformanceMonitor,
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
        gather_new_sample: Receiver<(ExampleWithScore, u32)>,
        sampling_signal_channel: Sender<Signal>,
        serial_sampling: bool,
        init_block: bool,
        min_ess: Option<f32>,
    ) -> BufferLoader {
        let new_examples = Arc::new(RwLock::new(None));
        let num_batch = (size + batch_size - 1) / batch_size;
        let gatherer = Gatherer::new(gather_new_sample, new_examples.clone(), size.clone());
        let mut buffer_loader = BufferLoader {
            size: size,
            batch_size: batch_size,
            num_batch: num_batch,

            examples: vec![],
            new_examples: new_examples,
            gatherer: gatherer,
            serial_sampling: serial_sampling,
            sampling_signal_channel: sampling_signal_channel,

            ess: 1.0,
            min_ess: min_ess.unwrap_or(0.0),
            curr_example: 0,
            sampling_pm: PerformanceMonitor::new(),
        };
        if !buffer_loader.serial_sampling {
            buffer_loader.sampling_signal_channel.send(Signal::START);
            buffer_loader.gatherer.run(false);
        }
        if init_block {
            buffer_loader.force_switch();
        }
        buffer_loader
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
        self.get_next_mut_batch(allow_switch)
    }

    pub fn get_next_batch_and_update(
        &mut self,
        allow_switch: bool,
        model: &Model
    ) -> &[ExampleInSampleSet] {
        let batch = self.get_next_mut_batch(allow_switch);
        update_scores(batch, model);
        batch
    }

    fn get_next_mut_batch(&mut self, allow_switch: bool) -> &mut [ExampleInSampleSet] {
        if allow_switch && !self.serial_sampling {
            self.try_switch();
        }
        self.curr_example += self.batch_size;
        if self.curr_example >= self.size {
            self.update_ess();
            self.curr_example = 0;
        }

        assert!(!self.examples.is_empty());
        let tail = min(self.curr_example + self.batch_size, self.size);
        &mut self.examples[self.curr_example..tail]
    }

    fn force_switch(&mut self) {
        self.sampling_pm.resume();
        if self.serial_sampling {
            self.sampling_signal_channel.send(Signal::START);
        }
        self.gatherer.run(true);
        if self.serial_sampling {
            self.sampling_signal_channel.send(Signal::STOP);
        }
        self.sampling_pm.pause();

        assert_eq!(self.try_switch(), true);
    }

    fn try_switch(&mut self) -> bool {
        self.sampling_pm.resume();
        let new_examples = {
            if let Ok(mut new_examples) = self.new_examples.try_write() {
                new_examples.take()
            } else {
                None
            }
        };
        let switched = {
            if new_examples.is_some() {
                self.examples = new_examples.unwrap()
                                            .into_iter()
                                            .map(|t| {
                                                let (a, s) = t;
                                                let w = get_weight(&a, 0.0);
                                                (a, (w, s.1))
                                            }).collect();
                self.curr_example = 0;
                debug!("switched-buffer, {}", self.examples.len());
                true
            } else {
                false
            }
        };
        self.sampling_pm.pause();
        switched
    }

    // ESS and others
    /// Get the estimate of the effective sample size of the current sample set.
    fn update_ess(&mut self) {
        let (sum_weights, sum_weight_squared) =
            self.examples.iter()
                         .map(|(_, (w, _))| { (w, w * w) })
                         .fold((0.0, 0.0), |acc, x| (acc.0 + x.0, acc.1 + x.1));
        self.ess = sum_weights.powi(2) / sum_weight_squared / (self.size as f32);
        debug!("loader-reset, {}", self.ess);
        if self.serial_sampling && self.ess < self.min_ess {
            debug!("serial (blocking) sampling started");
            self.force_switch();
            debug!("serial (blocking) sampling finished");
        }
    }

    pub fn get_sampling_duration(&self) -> f32 {
        self.sampling_pm.get_duration()
    }
}


/// Update the scores of the examples using `model`
fn update_scores(data: &mut [ExampleInSampleSet], model: &Model) {
    let model_size = model.len();
    data.par_iter_mut().for_each(|example| {
        let curr_weight = (example.1).0;
        let mut new_score = 0.0;
        for tree in model[((example.1).1)..model_size].iter() {
            new_score += tree.get_leaf_prediction(&example.0);
        }
        (*example).1 = (curr_weight * get_weight(&example.0, new_score), model_size);
    });
}


#[cfg(test)]
mod tests {
    use std::thread::sleep;
    use commons::channel;

    use std::time::Duration;
    use labeled_data::LabeledData;
    use commons::ExampleWithScore;
    use commons::Signal;
    use super::BufferLoader;


    #[test]
    fn test_buffer_loader() {
        let (sender, receiver) = channel::bounded(10, "gather-samples");
        let (signal_s, signal_r) = channel::bounded(10, "sampling-signal");
        let mut buffer_loader = BufferLoader::new(100, 10, receiver, signal_s, false, false, None);
        assert_eq!(signal_r.recv().unwrap(), Signal::START);
        sender.send((get_example(vec![0.0, 1.0, 2.0], 1.0), 100));
        sleep(Duration::from_millis(1000));
        for _ in 0..20 {
            let batch = buffer_loader.get_next_batch(true);
            assert_eq!(batch.len(), 10);
            assert_eq!((batch[0].1).0, 1.0);
            assert_eq!((batch[0].2).0, 1.0);
            assert_eq!((batch[9].1).0, 1.0);
            assert_eq!((batch[9].2).0, 1.0);
        }
        sender.send((get_example(vec![0.0, 1.0, 2.0], 2.0), 100));
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

    fn get_example(features: Vec<f32>, score: f32) -> ExampleWithScore {
        let label: i8 = -1;
        let example = LabeledData::new(features, label);
        (example, (score, 0))
    }
}
