use rand;
use rand::Rng;

use std::cmp::min;
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;

use commons::bins::Bins;
use commons::io::create_bufreader;
use commons::io::create_bufwriter;
use commons::io::read_k_labeled_data;
use commons::io::read_k_labeled_data_from_binary_file;
use commons::io::write_to_binary_file;
use commons::performance_monitor::PerformanceMonitor;
use labeled_data::LabeledData;

use super::super::Example;
use super::super::RawExample;
use super::super::TFeature;
use super::super::RawTFeature;

/// A naive file loader
#[derive(Debug)]
pub struct SerialStorage {
    filename: String,
    is_binary: bool,
    in_memory: bool,
    pub size: usize,
    feature_size: usize,
    positive: String,

    bytes_per_example: usize,
    binary_cons: Option<TextToBinHelper>,
    reader: BufReader<File>,
    memory_buffer: Vec<Example>,
    index: usize,
    bins: Vec<Bins>,

    head: usize,
    tail: usize,
}


impl SerialStorage {
    pub fn new(
        filename: String,
        size: usize,
        feature_size: usize,
        one_pass: bool,
        positive: String,
        bins: Option<Vec<Bins>>,
    ) -> SerialStorage {
        let reader = create_bufreader(&filename);
        let binary_cons = if one_pass {
            None
        } else {
            Some(TextToBinHelper::new(&filename))
        };
        debug!("Created a serial storage object for {}, capacity {}, feature size {}",
               filename, size, feature_size);
        SerialStorage {
            filename: filename,
            size: size.clone(),
            feature_size: feature_size,
            is_binary: false,
            in_memory: false,
            positive: positive,
            bytes_per_example: 0,

            binary_cons: binary_cons,
            reader: reader,
            memory_buffer: vec![],
            index: 0,
            bins: bins.unwrap_or(vec![]),

            head: 0,
            tail: 0,
        }
    }

    pub fn read_raw(&mut self, batch_size: usize) -> Vec<RawExample> {
        self.head = self.index;
        self.tail = min(self.index + batch_size, self.size);
        let true_batch_size = self.tail - self.head;

        // Raw data always read from disk
        let batch: Vec<RawExample> = {
            read_k_labeled_data(
                &mut self.reader, true_batch_size,
                0 as RawTFeature, self.feature_size, &self.positive,
            )
        };

        self.index = self.tail;
        self.try_reset(false /* not forcing */);
        batch
    }

    pub fn read(&mut self, batch_size: usize) -> Vec<Example> {
        self.head = self.index;
        self.tail = min(self.index + batch_size, self.size);
        let true_batch_size = self.tail - self.head;

        // Load from memory
        if self.in_memory {
            return self.memory_buffer[self.head..self.tail].to_vec();
        }
        // Load from disk
        let batch: Vec<Example> =
            if self.is_binary {
                read_k_labeled_data_from_binary_file(
                    &mut self.reader, true_batch_size, self.bytes_per_example)
            } else {
                let batch: Vec<RawExample> =
                    read_k_labeled_data(
                        &mut self.reader, true_batch_size,
                        0 as RawTFeature, self.feature_size, &self.positive,
                    );
                batch.into_iter().map(|data| {
                    let features: Vec<TFeature> =
                        data.feature.iter().enumerate()
                            .map(|(idx, val)| {
                                self.bins[idx].get_split_index(*val)
                            }).collect();
                    LabeledData::new(features, data.label)
                }).collect()
            };
        if let Some(ref mut cons) = self.binary_cons {
            batch.iter().for_each(|data| {
                cons.append_data(data);
            });
        }

        self.index = self.tail;
        self.try_reset(false /* not forcing */);
        batch
    }

    fn try_reset(&mut self, force: bool) {
        if self.index < self.size && !force {
            return;
        }

        if self.index >= self.size && self.binary_cons.is_some() {
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

        let mut pm = PerformanceMonitor::new();
        pm.start();
        self.try_reset(true /* force */);
        let num_batch = (self.size + batch_size - 1) / batch_size;
        for _ in 0..num_batch {
            let data = self.read(batch_size);
            pm.update(data.len());
            self.memory_buffer.extend(data);
            pm.write_log(&format!("load_{}", self.filename));
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
    pub fn new(original_filename: &String) -> TextToBinHelper {
        let filename = original_filename.clone() + "_bin";
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


#[allow(dead_code)]
fn gen_filename() -> String {
    let random_str =
        rand::thread_rng()
            .gen_ascii_chars()
            .take(6)
            .collect::<String>();
    String::from("data-") + random_str.as_str() + ".bin"
}
