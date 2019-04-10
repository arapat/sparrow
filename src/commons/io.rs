use bincode::serialize;
use bincode::deserialize;
use rayon::prelude::*;

use std::str::FromStr;
use std::fmt::Debug;

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Read;
use std::io::Write;
use std::io::Result;

use commons::Example;
use labeled_data::LabeledData;


pub fn create_bufreader(filename: &String) -> BufReader<File> {
    let f = File::open(filename).expect(&format!("Cannot open the file `{}`.", filename));
    BufReader::new(f)
}

pub fn create_bufwriter(filename: &String) -> BufWriter<File> {
    let f = File::create(filename).expect(&format!("Cannot create the file `{}`.", filename));
    BufWriter::new(f)
}

// TODO: create buffer for the file handler
pub fn raw_read_all(filename: &String) -> String {
    let mut file = File::open(filename).expect(
        &format!("File `{}` does not exist or cannot be read.", filename));
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect(&format!("Cannot read `{}`", filename));
    contents
}

// TODO: create buffer for the file handler
pub fn read_all(filename: &String) -> Vec<u8> {
    let mut content = Vec::new();
    let mut file = File::create(filename).expect(
        &format!("File `{}` does not exist or cannot be read.", filename));
    file.read_to_end(&mut content).unwrap();
    content
}

// TODO: create buffer for the file handler
pub fn write_all(filename: &String, content: &[u8]) -> Result<()> {
    let mut file = File::create(filename).expect(
        &format!("File `{}` does not exist or cannot be read.", filename));
    file.write_all(content)
}

pub fn read_k_lines(reader: &mut BufReader<File>, k: usize) -> Vec<String> {
    let mut ret: Vec<String> = vec![String::new(); k];
    for string in &mut ret {
        reader.read_line(string).unwrap();
    }
    ret
}

pub fn read_k_labeled_data<TFeature, TLabel>(
    reader: &mut BufReader<File>,
    k: usize,
    missing_val: TFeature,
    size: usize,
    positive: &String,
) -> Vec<LabeledData<TFeature, TLabel>>
where
    TFeature: FromStr + Clone + Send + Sync,
    TFeature::Err: Debug,
    TLabel: FromStr + Send + Sync,
    TLabel::Err: Debug,
{
    let lines = read_k_lines(reader, k);
    parse_libsvm(&lines, missing_val, size, positive)
}

pub fn read_k_labeled_data_from_binary_file(
    reader: &mut BufReader<File>,
    k: usize,
    data_size: usize
) -> Vec<Example> {
    let data: Vec<Vec<u8>> = (0..k).map(|_| {
        let mut buf: Vec<u8> = vec![0; data_size];
        reader.read_exact(&mut buf[..]).unwrap();
        buf
    }).collect();
    data.par_iter().map(|buf| {
        deserialize(&buf[..]).unwrap()
    }).collect()
}

pub fn write_to_binary_file(writer: &mut BufWriter<File>, data: &Example) -> usize {
    let serialized = serialize(data).unwrap();
    writer.write(serialized.as_ref()).unwrap();
    serialized.len()
}

#[inline]
fn parse_libsvm_one_line<TFeature, TLabel>(
    raw_string: &String,
    missing_val: TFeature,
    size: usize,
    positive: &String,
) -> LabeledData<TFeature, TLabel>
where
    TFeature: FromStr + Clone + Send + Sync,
    TFeature::Err: Debug,
    TLabel: FromStr + Send + Sync,
    TLabel::Err: Debug
{
    let mut numbers = raw_string.split_whitespace();
    let label: TLabel = {
        if numbers.next().unwrap() == *positive {
            "1".parse().unwrap()
        } else {
            "-1".parse().unwrap()
        }
    };
    let mut feature: Vec<TFeature> = vec![missing_val; size];
    numbers.map(|index_value| {
        let sep = index_value.find(':').unwrap();
        (
            index_value[..sep].parse().unwrap(),
            {
                if index_value[sep+1..].to_lowercase() == "nan" {
                    "0".parse().unwrap()  // TODO: comes up with a better placeholder for NAN
                } else {
                    index_value[sep+1..].parse().unwrap()
                }
            }
        )
    }).for_each(|(index, value): (usize, TFeature)| {
        feature[index] = value;
    });
    LabeledData::new(feature, label)
}

fn parse_libsvm<TFeature, TLabel>(
    raw_strings: &Vec<String>,
    missing_val: TFeature,
    size: usize,
    positive: &String,
) -> Vec<LabeledData<TFeature, TLabel>>
where
    TFeature: FromStr + Clone + Send + Sync,
    TFeature::Err: Debug,
    TLabel: FromStr + Send + Sync,
    TLabel::Err: Debug
{
    raw_strings.par_iter()
               .map(|s| parse_libsvm_one_line(&s, missing_val.clone(), size, positive))
               .collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_libsvm_one_line() {
        let raw_string = String::from("0 1:2 3:5 4:10");
        let label = -1;
        let feature = vec![0, 2, 0, 5, 10, 0];
        let labeled_data = LabeledData::new(feature, label);
        assert_eq!(parse_libsvm_one_line(&raw_string, 0, 6, &"1".to_string()), labeled_data);
    }

    #[test]
    fn test_parse_libsvm() {
        let raw_strings = vec![
            String::from("0 1:2 3:5 4:10"),
            String::from("1.2 1:3.0 2:10.0 4:10.0    5:20.0")
        ];
        let labeled_data = get_libsvm_answer();
        assert_eq!(parse_libsvm(&raw_strings, 0.0, 6, &"1.2".to_string()), labeled_data);
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
        assert_eq!(read_k_labeled_data(&mut f, 2, 0.0, 6, &"1.2".to_string()), labeled_data);
    }

    fn get_libsvm_file_path() -> String {
        String::from("tests/data/sample_libsvm.txt")
    }

    fn get_libsvm_answer() -> Vec<LabeledData<f32, f32>> {
        let label1 = -1.0;
        let feature1 = vec![0.0, 2.0, 0.0, 5.0, 10.0, 0.0];
        let label2 = 1.0;  // 1.2;
        let feature2 = vec![0.0, 3.0, 10.0, 0.0, 10.0, 20.0];
        vec![
            LabeledData::new(feature1, label1),
            LabeledData::new(feature2, label2)
        ]
    }
}
