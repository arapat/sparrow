use rand;
use rand::Rng;

use std::fs::File;
use std::io::BufWriter;

use commons::Example;
use data_loader::io::create_bufwriter;
use data_loader::io::write_to_binary_file;


#[derive(Debug)]
pub struct TextToBinHelper {
    filename: String,
    size: usize,
    bytes_per_example: usize,
    _writer: BufWriter<File>
}

impl TextToBinHelper {
    pub fn new() -> TextToBinHelper {
        let filename = gen_filename();
        let writer = create_bufwriter(&filename);
        TextToBinHelper {
            filename: filename,
            _writer: writer,
            size: 0,
            bytes_per_example: 0
        }
    }

    pub fn append_data(&mut self, data: &Example) {
        let bytes_size = write_to_binary_file(&mut self._writer, data);
        if self.bytes_per_example > 0 {
            assert_eq!(self.bytes_per_example, bytes_size);
        }
        self.bytes_per_example = bytes_size;
        self.size += 1;
    }

    pub fn get_content(&self) -> (String, usize, usize) {
        (self.filename.clone(), self.size, self.bytes_per_example)
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