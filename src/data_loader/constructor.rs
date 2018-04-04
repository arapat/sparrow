use rand;
use rand::Rng;

use std::fs::File;
use std::io::BufWriter;

use commons::Example;
use commons::get_symmetric_label;
use commons::is_positive;
use super::io::create_bufwriter;
use super::io::write_to_binary_file;


#[derive(Debug)]
pub struct Constructor {
    filename: String,
    scores: Vec<f32>,
    size: usize,

    bytes_per_example: usize,
    _writer: BufWriter<File>,

    num_positive: usize,
    num_negative: usize
}

impl Constructor {
    pub fn new(capacity: usize) -> Constructor {
        let filename = gen_filename();
        let writer = create_bufwriter(&filename);
        Constructor {
            filename: filename,
            scores: Vec::with_capacity(capacity),
            size: 0,

            bytes_per_example: 0,
            _writer: writer,

            num_positive: 0,
            num_negative: 0
        }
    }

    pub fn append_data(&mut self, data: &Example, score: f32) {
        let size = write_to_binary_file(&mut self._writer, data);
        if self.bytes_per_example > 0 {
            assert_eq!(self.bytes_per_example, size);
        }
        if is_positive(&get_symmetric_label(data)) {
            self.num_positive += 1;
        } else {
            self.num_negative += 1;
        }
        self.bytes_per_example = size;
        self.scores.push(score);
        self.size += 1;
    }

    pub fn get_content(self) -> (String, Vec<f32>, usize, usize) {
        debug!("A constructor is being consumed, which contains {} positive, {} negative.",
               self.num_positive, self.num_negative);
        (self.filename, self.scores, self.size, self.bytes_per_example)
    }

    pub fn get_filename(&self) -> String {
        self.filename.clone()
    }

    pub fn get_bytes_per_example(&self) -> usize {
        self.bytes_per_example.clone()
    }
}

fn gen_filename() -> String {
    let random_str =
        rand::thread_rng()
            .gen_ascii_chars()
            .take(6)
            .collect::<String>();
    String::from("data-") + random_str.as_str() + ".bin"
}
