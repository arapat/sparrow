use rand;
use rand::Rng;

use std::cmp::min;
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;

use commons::io::create_bufreader;
use commons::io::create_bufwriter;
use commons::io::read_k_labeled_data;
use commons::io::read_k_labeled_data_from_binary_file;
use commons::io::write_to_binary_file;

use super::super::Example;
use super::super::TLabel;

#[derive(Debug)]
pub struct SerialStorage {
    filename: String,
    is_binary: bool,
    in_memory: bool,
    size: usize,
    feature_size: usize,

    bytes_per_example: usize,
    binary_cons: Option<TextToBinHelper>,
    reader: BufReader<File>,
    memory_buffer: Vec<Example>,
    index: usize,
}


impl SerialStorage {
    pub fn new(
            filename: String,
            size: usize,
            feature_size: usize,
            is_binary: bool,
            bytes_per_example: Option<usize>,
    ) -> SerialStorage {
        assert!(!is_binary || bytes_per_example.is_some());

        let reader = create_bufreader(&filename);
        let binary_cons = if is_binary {
            None
        } else {
            Some(TextToBinHelper::new())
        };
        let unwrap_bytes_per_example =
            if let Some(t) = bytes_per_example {
                t
            } else {
                0
            };
        SerialStorage {
            filename: filename,
            size: size,
            feature_size: feature_size,
            is_binary: is_binary,
            in_memory: false,

            bytes_per_example: unwrap_bytes_per_example,
            binary_cons: binary_cons,
            reader: reader,
            memory_buffer: vec![],
            index: 0,
        }
    }

    pub fn read(&mut self, batch_size: usize) -> Vec<Example> {
        let head = self.index;
        let tail = min(self.index + batch_size, self.size);
        let true_batch_size = tail - head;
        self.index = tail;
        self.try_reset(false /* not forcing */);

        // Load from memory
        if self.in_memory {
            return self.memory_buffer[head..tail].to_vec();
        }
        // Load from disk
        let batch: Vec<Example> =
            if self.is_binary {
                read_k_labeled_data_from_binary_file(
                    &mut self.reader, true_batch_size, self.bytes_per_example)
            } else {
                read_k_labeled_data(
                    &mut self.reader, true_batch_size, 0 as TLabel, self.feature_size)
            };
        if let Some(ref mut cons) = self.binary_cons {
            batch.iter().for_each(|data| {
                cons.append_data(data);
            });
        }
        batch
    }

    fn try_reset(&mut self, force: bool) {
        if self.index < self.size && !force {
            return;
        }

        if self.index >= self.size && !self.is_binary {
            let cons = self.binary_cons.as_ref().unwrap();
            let orig_filename = self.filename.clone();
            let (filename, size, bytes_per_example) = cons.get_content();
            assert_eq!(size, self.size);
            self.is_binary = true;
            self.filename = filename;
            self.bytes_per_example = bytes_per_example;

            info!("Text-based loader for `{}` has been converted to Binary-based. \
                   Filename: {}, bytes_per_example: {}.",
                  orig_filename, self.filename, bytes_per_example);
         }

         self.binary_cons = None;
         self.index = 0;
         self.reader = create_bufreader(&self.filename);
    }

    #[allow(dead_code)]
    pub fn load_to_memory(&mut self, batch_size: usize) {
        info!("Load current file into the memory.");
        assert_eq!(self.in_memory, false);

        self.try_reset(true /* force */);
        let num_batch = (self.size + batch_size - 1) / batch_size;
        for _ in 0..num_batch {
            let data = self.read(batch_size);
            self.memory_buffer.extend(data);
        }
        self.in_memory = true;
        self.is_binary = true;
        info!("In-memory conversion finished.");
    }
}


#[derive(Debug)]
struct TextToBinHelper {
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