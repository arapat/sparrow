use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

use std::clone::Clone;
use std::str::FromStr;
use std::fmt::Debug;

use labeled_data::LabeledData;
use tree::Tree;

// TODO: use genetic types for reading data
type _TFeature = f32;
type _TLabel = f32;
type Examples = Vec<LabeledData<_TFeature, _TLabel>>;

#[derive(Debug)]
pub struct DataLoader<'a> {
    filename: String,
    size: usize,
    feature_size: usize,
    batch_size: usize,

    num_unique: usize,
    num_positive: usize,
    num_negative: usize,
    num_unique_positive: usize,
    num_unique_negative: usize,

    _full_scanned: bool,
    _num_batch: usize,
    _reader: BufReader<File>,
    _curr_loc: usize,
    _cursor: usize,
    _curr_batch: Examples,

    base_node: usize,
    scores_version: Vec<usize>,
    base_scores: Vec<f32>,
    scores: Vec<f32>,

    derive_from: Option<&'a DataLoader<'a>>
}

// TODO: write scores to disk
impl<'a> DataLoader<'a> {
    pub fn new(filename: String, size: usize, feature_size: usize, batch_size: usize,
               base_node: usize, scores: Vec<f32>) -> DataLoader<'a> {
        assert!(batch_size <= size);
        let reader = create_bufreader(&filename);
        let num_batch = size / batch_size + (size % batch_size as usize);
        DataLoader {
            filename: filename,
            size: size,
            feature_size: feature_size,
            batch_size: batch_size,

            num_unique: 0,
            num_positive: 0,
            num_negative: 0,
            num_unique_positive: 0,
            num_unique_negative: 0,

            _full_scanned: false,
            _num_batch: num_batch,
            _reader: reader,
            _curr_loc: 0,
            _cursor: 0,
            _curr_batch: vec![],

            base_node: 0,
            scores_version: vec![base_node; num_batch],
            base_scores: scores.clone(),
            scores: scores,

            derive_from: None
        }
    }

    pub fn set_source(&mut self, derive_from: &'a DataLoader<'a>) {
        self.derive_from = Some(derive_from);
    }

    pub fn get_next_batch(&mut self) -> &Examples {
        self._curr_loc = self._cursor;
        self._curr_batch = if (self._cursor + 1) * self.batch_size <= self.size {
            self._cursor += 1;
            read_k_labeled_data(&mut self._reader, self.batch_size, 0.0, self.feature_size)
        } else {
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
        &self._curr_batch
    }

    pub fn get_curr_batch_scores(&mut self, trees: &Vec<Tree>) -> (Vec<f32>, &[f32]) {
        let tree_head = self.scores_version[self._curr_loc];
        let tree_tail = trees.len();
        let head = self._curr_loc * self.batch_size;
        let tail = head + self._curr_batch.len();

        {
            let scores_region = &mut self.scores[head..tail];
            for tree in trees[tree_head..tree_tail].iter() {
                tree.add_prediction_to_score(&self._curr_batch, scores_region)
            }
            self.scores_version[self._curr_loc] = tree_tail;
        }

        (
            self.scores[head..tail]
                .iter()
                .zip(self.base_scores[head..tail].iter())
                .map(|(a, b)| a - b)
                .collect(),
            &self.scores[head..tail]
        )
    }

    fn set_bufrader(&mut self) {
        self._reader = create_bufreader(&self.filename);
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
        ) -> Vec<LabeledData<TFeature, TLabel>>
        where TFeature: FromStr + Clone, TFeature::Err: Debug,
              TLabel: FromStr, TLabel::Err: Debug {
    parse_libsvm(&read_k_lines(reader, k), missing_val, size)
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
        assert_eq!(read_k_labeled_data(&mut f, 2, 0.0, 6), labeled_data);
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
