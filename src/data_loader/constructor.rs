extern crate rand;

use std::fs::File;
use std::io::BufWriter;
use std::io::Write;

use self::rand::Rng;


pub struct Constructor {
    filename: String,
    scores: Vec<f32>,
    size: usize,

    _writer: BufWriter<File>
}

impl Constructor {
    pub fn new(capacity: usize) -> Constructor {
        let filename = gen_filename();
        let writer = create_bufwriter(&filename);
        Constructor {
            filename: filename,
            scores: Vec::with_capacity(capacity),
            size: 0,

            _writer: writer
        }
    }

    pub fn append_data(&mut self, data: &String, score: f32) {
        self._writer.write(data.as_bytes()).unwrap();
        self.scores.push(score);
        self.size += 1;
    }

    pub fn get_content(self) -> (String, Vec<f32>, usize) {
        (self.filename, self.scores, self.size)
    }
}


fn gen_filename() -> String {
    let random_str =
        rand::thread_rng()
            .gen_ascii_chars()
            .take(6)
            .collect::<String>();
    String::from("data-") + random_str.as_str() + ".txt"
}

fn create_bufwriter(filename: &String) -> BufWriter<File> {
    let f = File::create(filename).unwrap();
    BufWriter::new(f)
}
