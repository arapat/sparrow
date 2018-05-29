use std::fs::File;
use std::io::BufReader;

use commons::Example;
use commons::TLabel;
use data_loader::io::create_bufreader;
use data_loader::io::read_k_labeled_data;
use data_loader::io::read_k_labeled_data_from_binary_file;

use super::ExamplesReader;
use super::data_in_mem::DataInMem;
use super::text_to_bin_helper::TextToBinHelper;


#[derive(Debug)]
pub struct DataOnDisk {
    filename: String,
    is_binary: bool,
    binary_cons: Option<TextToBinHelper>,

    size: usize,
    feature_size: usize,
    bytes_per_example: usize,

    _reader: BufReader<File>
}


impl ExamplesReader for DataOnDisk {
    fn fetch(&mut self, batch_size: usize) -> Vec<Example> {
        let batch: Vec<Example> =
            if self.is_binary {
                read_k_labeled_data_from_binary_file(
                    &mut self._reader, batch_size, self.bytes_per_example)
            } else {
                read_k_labeled_data(&mut self._reader, batch_size, 0 as TLabel, self.feature_size)
            };
        if let Some(ref mut cons) = self.binary_cons {
            batch.iter().for_each(|data| {
                cons.append_data(data);
            });
        }
        batch
    }

    fn reset(&mut self) {
        if let Some(ref cons) = self.binary_cons {
            let orig_filename = self.filename.clone();
            let (filename, size, bytes_per_example) = cons.get_content();
            self.is_binary = true;
            self.filename = filename;
            self.bytes_per_example = bytes_per_example;
            info!("Text-based loader for `{}` has been converted to Binary-based. \
                   Filename: {}, bytes_per_example: {}.",
                  orig_filename, self.filename, bytes_per_example);
        }
        self.binary_cons = None;
        self._reader = create_bufreader(&self.filename);
    }
}


impl DataOnDisk {
    // TODO: add binary_constructor
    pub fn new(filename: String, is_binary: bool,
               size: usize, feature_size: usize, bytes_per_example: usize) -> DataOnDisk {
        let reader = create_bufreader(&filename);
        let binary_cons = if is_binary {
            None
        } else {
            Some(TextToBinHelper::new())
        };
        DataOnDisk {
            filename: filename,
            is_binary: is_binary,
            binary_cons: binary_cons,

            size: size,
            feature_size: feature_size,
            bytes_per_example: bytes_per_example,

            _reader: reader
        }
    }

    pub fn load_to_memory(&mut self, batch_size: usize) -> DataInMem {
        info!("Creating an in-memory version of the current loader.");
        let mut in_mem = DataInMem::new(self.size);
        let num_batch = (self.size + batch_size - 1) / batch_size;
        for _ in 0..num_batch {
            let data = self.fetch(batch_size);
            data.iter()
                .for_each(|d| {
                    in_mem.append(d);
                });
        }
        in_mem.release(false);
        info!("In-memory conversion finished.");
        in_mem
    }
}