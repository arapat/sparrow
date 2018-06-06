use rand::Rng;
use rand::thread_rng;

use std::ops::Range;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;
use chan::Receiver;

use std::thread::sleep;

use commons::ExampleInSampleSet;
use commons::ExampleWithScore;
use commons::Model;
use commons::performance_monitor::PerformanceMonitor;

use commons::get_weight;


#[derive(Debug)]
pub struct BufferLoader {
    size: usize,
    batch_size: usize,
    num_batch: usize,

    sampled_examples: Receiver<ExampleWithScore>,

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
    pub fn new(size: usize, batch_size: usize,
               sampled_examples: Receiver<ExampleWithScore>) -> BufferLoader {
        let num_batch = (size + batch_size - 1) / batch_size;
        BufferLoader {
            size: size,
            batch_size: batch_size,
            num_batch: num_batch,

            sampled_examples: sampled_examples,

            examples_in_use: vec![],
            examples_in_cons: Arc::new(RwLock::new(Vec::with_capacity(size))),
            is_cons_ready: Arc::new(RwLock::new(false)),

            ess: None,
            sum_weights: 0.0,
            sum_weight_squared: 0.0,

            _cursor: 0,
            _curr_range: (0..0),
            _scores_synced: false,
            performance: PerformanceMonitor::new()
        }
    }

    pub fn get_num_batches(&self) -> usize {
        self.num_batch
    }

    pub fn get_curr_batch(&self, is_scores_updated: bool) -> &[ExampleInSampleSet] {
        // scores must be updated unless is_scores_updated is not required.
        assert!(self._scores_synced || !is_scores_updated);
        let range = self._curr_range.clone();
        &self.examples_in_use[range]
    }

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

    pub fn update_scores(&mut self, model: &Model) {
        if self._scores_synced {
            return;
        }

        let model_size = model.len();
        let range = self._curr_range.clone();
        for example in self.examples_in_use[range].iter_mut() {
            let mut curr_score = (example.2).0;
            for tree in model[((example.2).1)..model_size].iter() {
                curr_score += tree.get_leaf_prediction(&example.0);
            }
            *example = (example.0.clone(), example.1, (curr_score, model_size));
        }
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

    fn try_switch(&mut self) {
        if let Ok(mut ready) = self.is_cons_ready.try_write() {
            if *ready {
                self.examples_in_use = self.examples_in_cons.read().unwrap().to_vec();
                self.examples_in_cons = Arc::new(RwLock::new(Vec::with_capacity(self.size)));
                self._cursor = 0;
                *ready = false;
            }
        }
    }

    // ESS and others
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
        debug!("loader-reset, {}", ess);
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
        loop {
            if let Ok(b) = is_cons_ready.try_read() {
                if *b == false {
                    break;
                } else {
                    sleep(Duration::from_secs(1));
                }
            }
        }
        if let Ok(mut examples) = examples_in_cons.write() {
            while examples.len() < capacity {
                let (example, (score, node)) = sampled_examples.recv().unwrap();
                examples.push((example, (score.clone(), node.clone()), (score, node)));
            }
            thread_rng().shuffle(&mut *examples);
        }
        let mut b = is_cons_ready.write().unwrap();
        *b = true;
    }
}