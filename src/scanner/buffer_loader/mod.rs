mod loader;

use rayon::prelude::*;
use std::cmp::min;
use std::sync::Arc;
use std::sync::RwLock;
use std::thread::sleep;
use std::time::Duration;

use SampleMode;
use commons::get_weight;
use commons::persistent_io::VersionedSampleModel;
use commons::performance_monitor::PerformanceMonitor;
use commons::ExampleWithScore;
use commons::ExampleInSampleSet;
use commons::Model;
use self::loader::Loader;


// LockedBuffer is set to None once it is read by the receiver
pub type LockedBuffer = Arc<RwLock<Option<VersionedSampleModel>>>;


/// Double-buffered sample set. It consists of two buffers stores in memory. One of the
/// buffer is used for providing examples to the boosting algorithm, meanwhile the other
/// would receive new examples from the sampler module. Once the other buffer is filled,
/// the functions of the two buffers switch so that the sample set used for training
/// would be updated periodically.
///
/// It might take several scans over the sample set before the new sample set is ready.
/// Thus it is very likely that the examples in the sample set would be accessed multiple times.
pub struct BufferLoader {
    pub size: usize,
    batch_size: usize,
    num_batch: usize,

    examples: Vec<ExampleInSampleSet>,
    pub base_model: Model,
    pub base_model_sig: String,
    pub current_version: usize,
    pub new_buffer: LockedBuffer,
    loader: Loader,
    pub sample_mode: SampleMode,

    pub ess: f32,
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
        sample_mode: SampleMode,
        sleep_duration: usize,
        min_ess: f32,
        exp_name: String,
    ) -> BufferLoader {
        let new_buffer = Arc::new(RwLock::new(None));
        let num_batch = (size + batch_size - 1) / batch_size;
        // Strata -> BufferLoader
        let loader = Loader::new(new_buffer.clone(), sleep_duration, exp_name);
        let buffer_loader = BufferLoader {
            size: size,
            batch_size: batch_size,
            num_batch: num_batch,

            examples: vec![],
            base_model: Model::new(1),
            base_model_sig: "default".to_string(),
            current_version: 0,
            new_buffer: new_buffer,
            loader: loader,
            sample_mode: sample_mode.clone(),

            ess: 0.0,
            min_ess: min_ess,
            curr_example: 0,
            sampling_pm: PerformanceMonitor::new(),
        };
        buffer_loader.loader.run(sample_mode.clone());
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
    pub fn get_next_batch(&mut self, allow_switch: bool) -> (&[ExampleInSampleSet], bool) {
        let (examples, switched) = self.get_next_mut_batch(allow_switch);
        (&*examples, switched)
    }

    pub fn get_next_batch_and_update(
        &mut self,
        allow_switch: bool,
        model: &Model,
    ) -> (&[ExampleInSampleSet], bool) {
        let (batch, switched) = self.get_next_mut_batch(allow_switch);
        update_scores(batch, model);
        (batch, switched)
    }

    fn get_next_mut_batch(&mut self, allow_switch: bool) -> (&mut [ExampleInSampleSet], bool) {
        let mut switched = false;
        // if self.ess <= self.min_ess && allow_switch {
        if allow_switch {
            switched = self.try_switch();
        }
        self.curr_example += self.batch_size;
        if self.curr_example >= self.size {
            self.update_ess();
            self.curr_example = 0;
        }

        assert!(!self.examples.is_empty());
        let tail = min(self.curr_example + self.batch_size, self.size);
        (&mut self.examples[self.curr_example..tail], switched)
    }

    pub fn try_switch(&mut self) -> bool {
        self.sampling_pm.resume();
        let new_buffer = self.new_buffer.try_write();
        if new_buffer.is_err() {
            self.sampling_pm.pause();
            return false;
        }
        let mut new_buffer = new_buffer.unwrap();
        if new_buffer.is_none() {
            self.sampling_pm.pause();
            return false;
        }

        let (new_version, new_examples, new_model, model_sig): VersionedSampleModel =
            new_buffer.take().unwrap();
        drop(new_buffer);
        let old_version = self.current_version;
        self.current_version = new_version;
        self.examples = set_init_weight(new_examples);
        self.base_model = new_model;
        self.base_model_sig = model_sig;
        self.curr_example = 0;

        self.update_ess();
        debug!("scanner, switched-buffer, {}, {}, {}",
                old_version, self.current_version, self.examples.len());
        self.sampling_pm.pause();
        true
    }

    // ESS and others

    // Check ess, if it is too small, block the thread until a new sample is received
    pub fn check_ess_blocking(&mut self) {
        let mut timer = PerformanceMonitor::new();
        let mut last_report_time = 0.0;
        timer.start();
        while self.ess < self.min_ess && !self.try_switch() {
            if timer.get_duration() - last_report_time > 10.0 {
                last_report_time = timer.get_duration();
                debug!("loader, blocking, {}, {}, {}, {}, {}",
                        last_report_time, self.ess, self.min_ess, self.base_model_sig,
                        self.current_version);
            }
            sleep(Duration::from_secs(2));
        }
    }

    /// Get the estimate of the effective sample size of the current sample set.
    fn update_ess(&mut self) {
        let (sum_weights, sum_weight_squared): (f32, f32) =
            self.examples.iter()
                         .map(|(_, (w, _, _, _))| { (w, w * w) })
                         .fold((0.0, 0.0), |acc, x| (acc.0 + x.0, acc.1 + x.1));
        self.ess = sum_weights.powi(2) / sum_weight_squared / (self.size as f32);
        debug!("loader-reset, {}", self.ess);
    }

    pub fn reset_scores(&mut self) {
        debug!("buffer-loader, reset all examples");
        reset_scores(&mut self.examples, &self.base_model);
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}


/// Update the scores of the examples using `model`
fn update_scores(data: &mut [ExampleInSampleSet], model: &Model) {
    data.par_iter_mut().for_each(|example| {
        let (_curr_weight, curr_score, mut curr_version, mut base_version) = example.1;
        if base_version != model.base_version {
            curr_version = base_version;  // reset score
            base_version = model.base_version;
        }
        let (new_score, (new_version, _)) = model.get_prediction(&example.0, curr_version);
        let updated_score = new_score + curr_score;
        (*example).1 = (
            get_weight(&example.0, updated_score), updated_score, new_version, base_version);
    });
}


/// Reset scores to the base model
fn reset_scores(data: &mut [ExampleInSampleSet], base_model: &Model) {
    let base_version = base_model.base_version;
    data.par_iter_mut().for_each(|example| {
        (*example).1 = (get_weight(&example.0, 0.0), 0.0, base_version, base_version);
    });
}



/// Set initial weights to the samples
fn set_init_weight(examples: Vec<ExampleWithScore>) -> Vec<ExampleInSampleSet> {
    examples.into_iter()
            .map(|t| {
                let (example, (_score, model_size)) = t;
                // sampling weights are ignored
                let w = get_weight(&example, 0.0);
                (example.clone(), (w, 0.0, model_size, model_size))
            }).collect()
}



#[cfg(test)]
mod tests {
    use std::thread::sleep;
    use commons::channel;
    use commons::io::delete_s3;

    use std::time::Duration;
    use labeled_data::LabeledData;
    use commons::ExampleWithScore;
    use commons::Signal;
    use super::BufferLoader;
    use TFeature;

    use super::io::REGION;
    use super::io::BUCKET;
    use super::io::S3_PATH;
    use super::io::FILENAME;

    #[test]
    fn test_buffer_loader_memory() {
        test_buffer_loader("memory");
    }

    #[test]
    fn test_buffer_loader_local() {
        test_buffer_loader("local");
    }

    #[test]
    fn test_buffer_loader_s3() {
        delete_s3(REGION, BUCKET, S3_PATH, FILENAME);
        test_buffer_loader("s3");
    }

    fn test_buffer_loader(mode: &str) {
        let (sender, receiver) = channel::bounded(10, "gather-samples");
        let (signal_s, signal_r) = channel::bounded(10, "sampling-signal");
        let mut buffer_loader = BufferLoader::new(
            100, 10, mode.to_string(), receiver, signal_s, 1, false, None, "both".to_string());
        assert_eq!(signal_r.recv().unwrap(), Signal::START);
        sender.send(((get_example(vec![0, 1, 2], -1, 1.0), 100), 100));
        while !buffer_loader.try_switch() {
            sleep(Duration::from_millis(1000));
        }
        for _ in 0..20 {
            let batch = buffer_loader.get_next_batch(true);
            assert_eq!(batch.len(), 10);
            assert_eq!((batch[0].1).0, 1.0);
            assert_eq!((batch[0].0).label, -1);
            assert_eq!((batch[9].1).0, 1.0);
            assert_eq!((batch[9].0).label, -1);
        }
        sender.send(((get_example(vec![0, 1, 2], 1, 2.0), 100), 100));
        while !buffer_loader.try_switch() {
            sleep(Duration::from_millis(1000));
        }
        for _ in 0..10 {
            let batch = buffer_loader.get_next_batch(true);
            assert_eq!(batch.len(), 10);
            assert_eq!((batch[0].1).0, 1.0);
            assert_eq!((batch[0].0).label, 1);
            assert_eq!((batch[9].1).0, 1.0);
            assert_eq!((batch[9].0).label, 1);
        }
    }

    fn get_example(features: Vec<TFeature>, label: i8, score: f32) -> ExampleWithScore {
        let example = LabeledData::new(features, label);
        (example, (score, 0))
    }
}
