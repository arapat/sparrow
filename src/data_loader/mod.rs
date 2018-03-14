extern crate rand;

mod constructor;

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

use std::clone::Clone;
use std::str::FromStr;
use std::fmt::Debug;

use self::rand::Rng;

use commons::max;
use commons::get_weight;
use commons::get_weights;
use commons::is_positive;
use commons::Example;
use commons::Model;
use commons::PerformanceMonitor;
use labeled_data::LabeledData;
use self::constructor::Constructor;


#[derive(Debug)]
pub struct DataLoader {
    filename: String,
    size: usize,
    feature_size: usize,
    batch_size: usize,
    num_batch: usize,

    num_positive: usize,
    num_negative: usize,
    sum_weights: f32,
    sum_weight_squared: f32,
    ess: Option<f32>,

    _reader: BufReader<File>,
    _curr_loc: usize,
    _cursor: usize,
    _curr_batch: Vec<Example>,
    _curr_batch_str: Vec<String>,
    _scores_synced: bool,

    base_node: usize,
    scores_version: Vec<usize>,
    base_scores: Vec<f32>,
    scores: Vec<f32>,
    relative_scores: Vec<f32>,

    performance: PerformanceMonitor
}

// TODO: write scores to disk
impl DataLoader {
    pub fn new(filename: String, size: usize, feature_size: usize, batch_size: usize,
               base_node: usize, scores: Vec<f32>) -> DataLoader {
        assert!(batch_size <= size);
        let reader = create_bufreader(&filename);
        let num_batch = size / batch_size + (size % batch_size as usize);
        let relative_scores = vec![0.0; size];
        debug!(
            "New DataLoader is created for `{}` \
            (size = {}, feature_size = {}, batch_size = {}, base_node = {})",
            filename, size, feature_size, batch_size, base_node
        );
        DataLoader {
            filename: filename,
            size: size,
            feature_size: feature_size,
            batch_size: batch_size,
            num_batch: num_batch,

            num_positive: 0,
            num_negative: 0,
            sum_weights: 0.0,
            sum_weight_squared: 0.0,
            ess: None,

            _reader: reader,
            _curr_loc: 0,
            _cursor: 0,
            _curr_batch: vec![],
            _curr_batch_str: vec![],
            _scores_synced: false,

            base_node: 0,
            scores_version: vec![base_node; num_batch],
            base_scores: scores.clone(),
            scores: scores,
            relative_scores: relative_scores,

            performance: PerformanceMonitor::new()
        }
    }

    pub fn from_scratch(filename: String, size: usize, feature_size: usize,
                        batch_size: usize) -> DataLoader {
        DataLoader::new(filename, size, feature_size, batch_size, 0, vec![0.0; size])
    }

    pub fn from_constructor(&self, constructor: Constructor, base_node: usize) -> DataLoader {
        let (filename, scores, size): (String, Vec<f32>, usize) = constructor.get_content();
        let mut new_loader = DataLoader::new(
            filename,
            size,
            self.feature_size,
            self.batch_size,
            base_node,
            scores
        );
        new_loader.scores.shrink_to_fit();
        new_loader
    }

    pub fn get_feature_size(&self) -> usize {
        self.feature_size
    }

    pub fn get_ess(&self) -> Option<f32> {
        self.ess
    }

    pub fn get_curr_batch(&self) -> &Vec<Example> {
        &self._curr_batch
    }

    pub fn get_curr_batch_str(&self) -> &Vec<String> {
        &self._curr_batch_str
    }

    pub fn get_relative_scores(&self) -> &[f32] {
        assert!(self._scores_synced);
        &self.relative_scores.as_slice()
    }

    pub fn get_absolute_scores(&self) -> &[f32] {
        assert!(self._scores_synced);
        let head = self._curr_loc * self.batch_size;
        let tail = head + self._curr_batch.len();
        &self.scores[head..tail]
    }

    pub fn get_num_examples(&self) -> usize {
        self.size
    }

    pub fn get_num_batches(&self) -> usize {
        self.num_batch
    }

    pub fn fetch_next_batch(&mut self) {
        let mut loader_reset = false;
        self._curr_loc = self._cursor;
        let (batch, batch_str) = if (self._cursor + 1) * self.batch_size <= self.size {
            self._cursor += 1;
            read_k_labeled_data(&mut self._reader, self.batch_size, 0.0, self.feature_size)
        } else {
            loader_reset = true;
            let tail_remains = self.size - self._cursor * self.batch_size;
            self._cursor = 0;
            if tail_remains > 0 {
                let ret = read_k_labeled_data(&mut self._reader, tail_remains, 0.0, self.feature_size);
                self.set_bufrader();
                ret
            } else {
                self.set_bufrader();
                self._cursor += 1;
                read_k_labeled_data(&mut self._reader, self.batch_size, 0.0, self.feature_size)
            }
        };
        self._curr_batch = batch;
        self._curr_batch_str = batch_str;
        self._scores_synced = false;

        if loader_reset {
            // update ESS
            let count = self.num_positive + self.num_negative;
            let ess = self.sum_weights.powi(2) / self.sum_weight_squared / (count as f32);
            self.ess = Some(ess);
            self.num_positive = 0;
            self.num_negative = 0;
            self.sum_weights = 0.0;
            self.sum_weight_squared = 0.0;
            info!("ESS is updated: {}", ess);
            debug!("Loader have reset.");
        }

        self.performance.update(self._curr_batch.len());
        let (count, duration, speed) = self.performance.get_performance();
        if count >= 10000 {
            debug!("Loader speed is {}.", speed);
            self.performance.start();
        }
    }

    pub fn fetch_scores(&mut self, trees: &Model) {
        if self._scores_synced {
            return;
        }

        let tree_head = self.scores_version[self._curr_loc];
        let tree_tail = trees.len();
        let head = self._curr_loc * self.batch_size;
        let tail = head + self._curr_batch.len();

        {
            let scores_region = &mut self.scores[head..tail];
            for tree in trees[tree_head..tree_tail].iter() {
                tree.add_prediction_to_score(&self._curr_batch, scores_region)
            }
        }
        self.relative_scores = self.scores[head..tail]
                                   .iter()
                                   .zip(self.base_scores[head..tail].iter())
                                   .map(|(a, b)| a - b)
                                   .collect();
        self.scores_version[self._curr_loc] = tree_tail;
        self._scores_synced = true;
        self.update_stats_for_ess();
    }

    fn update_stats_for_ess(&mut self) {
        let mut num_positive       = 0;
        let mut num_negative       = 0;
        let mut sum_weights        = 0.0;
        let mut sum_weight_squared = 0.0;
        self._curr_batch
            .iter()
            .zip(self.relative_scores.iter())
            .for_each(|(data, score)| {
                if is_positive(data.get_label()) {
                    num_positive += 1;
                } else {
                    num_negative += 1;
                }
                let w = get_weight(data, *score);
                sum_weights += w;
                sum_weight_squared += w * w;
            });
        self.num_positive       += num_positive;
        self.num_negative       += num_negative;
        self.sum_weights        += sum_weights;
        self.sum_weight_squared += sum_weight_squared;
    }

    fn set_bufrader(&mut self) {
        self._reader = create_bufreader(&self.filename);
    }

    // TODO: implement stratified sampling version
    pub fn sample(&mut self, trees: &Model, sample_ratio: f32) -> DataLoader {
        debug!("Sampling started. Sample ratio is {}. Data size is {}.", sample_ratio, self.size);
        let (interval, size) = self.get_estimated_interval_and_size(trees, sample_ratio);
        debug!("Sample size is estimated to be {}.", size);

        let mut sum_weights = (rand::thread_rng().gen::<f32>()) * interval;
        let mut constructor = Constructor::new(size);
        for _ in 0..self.num_batch {
            self.fetch_next_batch();
            self.fetch_scores(trees);
            let data = self.get_curr_batch();
            self.get_absolute_scores()
                .iter()
                .zip(data.iter().zip(self._curr_batch_str.iter()))
                .for_each(|(score, (data, data_str))| {
                    let w = get_weight(data, *score);
                    let next_sum_weight = sum_weights + w;
                    let num_copies =
                        (next_sum_weight / interval) as usize - (sum_weights / interval) as usize;
                    if num_copies > 0 {
                        constructor.append_data(data_str, score * (num_copies as f32));
                    }
                    sum_weights = next_sum_weight;
                });
        }
        let ret = self.from_constructor(constructor, trees.len());
        debug!("Sampling finished. Sample size is {}.", ret.get_num_examples());
        ret
    }

    fn get_estimated_interval_and_size(&mut self, trees: &Model, sample_ratio: f32) -> (f32, usize) {
        let mut sum_weights = 0.0;
        let mut max_weight = 0.0;
        for _ in 0..self.num_batch {
            self.fetch_next_batch();
            self.fetch_scores(trees);
            let data = self.get_curr_batch();
            let scores = self.get_absolute_scores();
            let ws = get_weights(&data, &scores);
            ws.iter().for_each(|w| {
                sum_weights += w;
                max_weight = max(max_weight, *w);
            });
        }
        let sample_size = (sample_ratio * self.size as f32) as usize + 1;
        let interval = sum_weights / (sample_size as f32);
        // TODO: log max_repeat
        let _max_repeat = max_weight / interval;
        (interval, sample_size + 10)
    }
}


pub fn create_bufreader(filename: &String) -> BufReader<File> {
    let f = File::open(filename).unwrap();
    BufReader::new(f)
}

pub fn read_k_lines(reader: &mut BufReader<File>, k: usize) -> Vec<String> {
    let mut ret: Vec<String> = vec![String::new(); k];
    for mut string in &mut ret {
        reader.read_line(string).unwrap();
    }
    ret
}

pub fn read_k_labeled_data<TFeature, TLabel>(
            reader: &mut BufReader<File>, k: usize, missing_val: TFeature, size: usize
        ) -> (Vec<LabeledData<TFeature, TLabel>>, Vec<String>)
        where TFeature: FromStr + Clone, TFeature::Err: Debug,
              TLabel: FromStr, TLabel::Err: Debug {
    let lines = read_k_lines(reader, k);
    (parse_libsvm(&lines, missing_val, size), lines)
}

pub fn parse_libsvm_one_line<TFeature, TLabel>(
            raw_string: &String, missing_val: TFeature, size: usize
        ) -> LabeledData<TFeature, TLabel>
        where TFeature: FromStr + Clone, TFeature::Err: Debug,
              TLabel: FromStr, TLabel::Err: Debug {
    let mut numbers = raw_string.split_whitespace();
    let label = numbers.next().unwrap().parse::<TLabel>().unwrap();
    let mut feature: Vec<TFeature> = vec![missing_val; size];
    for mut index_value in numbers.map(|s| s.split(':')) {
        let index = index_value.next().unwrap().parse::<usize>().unwrap();
        let value = index_value.next().unwrap().parse::<TFeature>().unwrap();
        feature[index] = value;
    }
    LabeledData::new(feature, label)
}

pub fn parse_libsvm<TFeature, TLabel>(raw_strings: &Vec<String>, missing_val: TFeature, size: usize)
        -> Vec<LabeledData<TFeature, TLabel>>
        where TFeature: FromStr + Clone, TFeature::Err: Debug,
              TLabel: FromStr, TLabel::Err: Debug {
    raw_strings.iter().map(|s| parse_libsvm_one_line(&s, missing_val.clone(), size)).collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_libsvm_one_line() {
        let raw_string = String::from("0 1:2 3:5 4:10");
        let label = 0;
        let feature = vec![0, 2, 0, 5, 10, 0];
        let labeled_data = LabeledData::new(feature, label);
        assert_eq!(parse_libsvm_one_line(&raw_string, 0, 6), labeled_data);
    }

    #[test]
    fn test_parse_libsvm() {
        let raw_strings = vec![
            String::from("0 1:2 3:5 4:10"),
            String::from("1.2 1:3.0 2:10.0 4:10.0    5:20.0")
        ];
        let labeled_data = get_libsvm_answer();
        assert_eq!(parse_libsvm(&raw_strings, 0.0, 6), labeled_data);
    }

    #[test]
    fn test_read_file() {
        let raw_strings = vec![
            String::from("0 1:2 3:5 4:10\n"),
            String::from("1.2 1:3.0 2:10.0 4:10.0    5:20.0\n")
        ];
        let mut f = create_bufreader(&get_libsvm_file_path());
        let from_file = read_k_lines(&mut f, 2);
        assert_eq!(from_file, raw_strings);
    }

    #[test]
    fn test_read_libsvm() {
        let mut f = create_bufreader(&get_libsvm_file_path());
        let labeled_data = get_libsvm_answer();
        assert_eq!(read_k_labeled_data(&mut f, 2, 0.0, 6).0, labeled_data);
    }

    fn get_libsvm_file_path() -> String {
        String::from("tests/data/sample_libsvm.txt")
    }

    fn get_libsvm_answer() -> Vec<LabeledData<f32, f32>> {
        let label1 = 0.0;
        let feature1 = vec![0.0, 2.0, 0.0, 5.0, 10.0, 0.0];
        let label2 = 1.2;
        let feature2 = vec![0.0, 3.0, 10.0, 0.0, 10.0, 20.0];
        vec![
            LabeledData::new(feature1, label1),
            LabeledData::new(feature2, label2)
        ]
    }
}
