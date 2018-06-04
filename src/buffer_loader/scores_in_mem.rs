use commons::performance_monitor::PerformanceMonitor;


#[derive(Debug)]
pub struct ScoresInMem {
    size: usize,
    batch_size: usize,
    num_batch: usize,

    base_node: usize,
    scores_version: Vec<usize>,
    base_scores: Vec<f32>,
    scores: Vec<f32>,
    relative_scores: Vec<f32>,

    _curr_loc: usize,
    _curr_batch_size: usize,
    _scores_synced: bool,

    performance: PerformanceMonitor
}


impl ScoresInMem {
    pub fn new(size: usize, batch_size: usize) -> ScoresInMem {
        let num_batch = (size + batch_size - 1) / batch_size;
        ScoresInMem {
            size: size,
            batch_size: batch_size.clone(),
            num_batch: num_batch,

            base_node: 0,
            scores_version: vec![0; num_batch],
            base_scores: vec![0.0; size],
            scores: vec![0.0; size],
            relative_scores: vec![0.0; size],

            _curr_loc: 0,
            _curr_batch_size: batch_size,
            _scores_synced: false,

            performance: PerformanceMonitor::new()
        }
    }

    pub fn set_base_scores(&mut self, base_node: usize, base_scores: Vec<f32>) {
        self.base_node = base_node;
        self.scores_version = vec![base_node; self.num_batch];
        self.base_scores = base_scores.clone();
        self.scores = base_scores;
        self.relative_scores = vec![0.0; self.size];
    }

    pub fn get_relative_scores(&self) -> &[f32] {
        assert!(self._scores_synced);
        &self.relative_scores.as_slice()
    }

    pub fn get_absolute_scores(&self) -> &[f32] {
        assert!(self._scores_synced);
        let head = self._curr_loc * self.batch_size;
        let tail = head + self._curr_batch_size;
        &self.scores[head..tail]
    }

    pub fn unset_scores_sync(&mut self, curr_loc: usize) {
        self._scores_synced = false;
        self._curr_loc = curr_loc;
    }

    pub fn get_scores_version(&self) -> usize {
        self.scores_version[self._curr_loc]
    }

    pub fn update_scores(&mut self, updated_scores: &Vec<f32>, tree_len: usize) {
        if self._scores_synced {
            return;
        }
        self.performance.resume();

        let head = self._curr_loc * self.batch_size;
        let tail = head + updated_scores.len();
        self.scores[head..tail].iter_mut()
            .zip(updated_scores.iter())
            .for_each(|(a, b)| *a += b);
        self.relative_scores = self.scores[head..tail].iter()
                                   .zip(self.base_scores[head..tail].iter())
                                   .map(|(a, b)| a - b)
                                   .collect();
        self.scores_version[self._curr_loc] = tree_len;
        self._scores_synced = true;
        self._curr_batch_size = updated_scores.len();

        self.performance.update(tail - head);
        self.performance.pause();
        // self.report_timer(self.performance, "loader-scoring-speed");
    }

    pub fn reset_scores(&mut self) {
        self.scores_version = vec![0; self.num_batch];
        self.scores = vec![0.0; self.scores.len()];
    }
}