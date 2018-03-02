use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

use std::clone::Clone;
use std::str::FromStr;
use std::fmt::Debug;

use labeled_data::LabeledData;

type _TFeature = f32;
type _TLabel = f32;

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
    _reader: BufReader<File>,
    _cursor: usize,

    base_node: usize,
    base_scores: Vec<f32>,
    scores: Vec<f32>,

    source: Option<&'a DataLoader<'a>>
}

// TODO: use genetic types for reading data
impl<'a> DataLoader<'a> {
    pub fn new(filename: String, size: usize, feature_size: usize, batch_size: usize)
            -> DataLoader<'a> {
        let reader = create_bufreader(&filename);
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
            _reader: reader,
            _cursor: 0,

            base_node: 0,
            base_scores: vec![],
            scores: vec![],

            source: None
        }
    }

    pub fn set_source(&mut self, source: &'a DataLoader<'a>) {
        self.source = Some(source);
    }

    pub fn get_next_batch(&mut self) -> Vec<LabeledData<_TFeature, _TLabel>> {
        if self._cursor + self.batch_size <= self.size {
            self._cursor = self._cursor + self.batch_size;
            read_k_labeled_data(&mut self._reader, self.batch_size, 0.0, self.feature_size)
        } else {
            let from_tail = self.size - self._cursor;
            let from_head = self.batch_size - from_tail;
            self._cursor = from_head;
            if from_tail > 0 {
                let mut a = read_k_labeled_data(&mut self._reader, from_tail, 0.0, self.feature_size);
                self.set_bufrader();
                let b = read_k_labeled_data(&mut self._reader, from_head, 0.0, self.feature_size);
                a.extend(b);
                a
            } else {
                self.set_bufrader();
                read_k_labeled_data(&mut self._reader, from_head, 0.0, self.feature_size)
            }
        }
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
