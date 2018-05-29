use rand;
use rand::Rng;

use std::fs::File;
use std::io::BufWriter;

use commons::Example;
use commons::get_symmetric_label;
use commons::is_positive;
use super::io::create_bufwriter;
use super::io::write_to_binary_file;
use super::examples_in_mem::Examples;


#[derive(Debug)]
pub struct Constructor {
    filename: String,
    examples: Option<Examples>,
    scores: Vec<f32>,
    size: usize,

    bytes_per_example: usize,
    _writer: Option<BufWriter<File>>,

    num_positive: usize,
    num_negative: usize
}

impl Constructor {
    pub fn new(capacity: usize, in_memory: bool) -> Constructor {
        let mut writer = None;
        let mut examples = None;
        let filename = if in_memory {
            examples = Some(Examples::new(capacity));
            String::from("__in_memory__")
        } else {
            let filename = gen_filename();
            writer = Some(create_bufwriter(&filename));
            filename
        };
        Constructor {
            filename: filename,
            examples: examples,
            scores: Vec::with_capacity(capacity),
            size: 0,

            bytes_per_example: 0,
            _writer: writer,

            num_positive: 0,
            num_negative: 0
        }
    }

    pub fn append_data(&mut self, data: &Example, score: f32) {
        match self.examples {
            None => {
                let size = if let Some(ref mut writer) = self._writer {
                    write_to_binary_file(writer, data)
                } else {
                    assert!(false);
                    0
                };
                if self.bytes_per_example > 0 {
                    assert_eq!(self.bytes_per_example, size);
                }
                self.bytes_per_example = size;
            },
            Some(ref mut ex) => {
                ex.append(data);
            }
        };

        if is_positive(&get_symmetric_label(data)) {
            self.num_positive += 1;
        } else {
            self.num_negative += 1;
        }
        self.scores.push(score);
        self.size += 1;
    }

    pub fn get_content(mut self) -> (String, Option<Examples>, Vec<f32>, usize, usize) {
        debug!("constructor-consumed, {}, {}", self.num_positive, self.num_negative);
        if let Some(ref mut examples) = self.examples {
            examples.shuffle();
        }
        (self.filename, self.examples, self.scores, self.size, self.bytes_per_example)
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
