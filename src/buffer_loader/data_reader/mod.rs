pub mod text_to_bin_helper;
pub mod data_in_mem;
pub mod data_on_disk;

use commons::Example;
use commons::performance_monitor::PerformanceMonitor;

use self::data_in_mem::DataInMem;
use self::data_on_disk::DataOnDisk;


pub trait ExamplesReader {
    fn fetch(&mut self, batch_size: usize) -> Vec<Example>;
    fn reset(&mut self);
}


#[derive(Debug)]
pub enum ActualReader {
    InMem(DataInMem),
    OnDisk(DataOnDisk)
}


#[derive(Debug)]
pub struct DataReader {
    size: usize,
    batch_size: usize,
    num_batch: usize,

    _cursor: usize,
    _reader: ActualReader,
    _curr_batch: Vec<Example>,

    performance: PerformanceMonitor
}


impl DataReader {
    pub fn new(size: usize, batch_size: usize, reader: ActualReader) -> DataReader {
        let num_batch = (size + batch_size - 1) / batch_size;
        DataReader {
            size: size,
            batch_size: batch_size,
            num_batch: num_batch,

            _cursor: 0,
            _reader: reader,
            _curr_batch: vec![],

            performance: PerformanceMonitor::new()
        }
    }

    pub fn get_num_batches(&self) -> usize {
        self.num_batch
    }

    pub fn get_curr_batch(&self) -> &Vec<Example> {
        &self._curr_batch
    }

    pub fn fetch_next_batch(&mut self) -> (usize, usize, bool) {
        self.performance.resume();

        let (curr_loc, batch_size) = self.get_next_batch_size();
        let reset = self._cursor == 0;
        self._curr_batch = match self._reader {
            ActualReader::InMem(ref mut a) => fetch_and_reset(a, batch_size, reset),
            ActualReader::OnDisk(ref mut b) => fetch_and_reset(b, batch_size, reset)
        };

        self.performance.update(self._curr_batch.len());
        self.performance.pause();

        (curr_loc, batch_size, reset)
    }

    fn get_next_batch_size(&mut self) -> (usize, usize) {
        let curr_loc = self._cursor;
        let batch_size = if (self._cursor + 1) * self.batch_size < self.size {
            self._cursor += 1;
            self.batch_size
        } else {
            let tail_remains = self.size - self._cursor * self.batch_size;
            self._cursor = 0;
            tail_remains
        };
        (curr_loc, batch_size)
    }

    pub fn load_to_memory(&mut self) {
        let new_reader = if let ActualReader::OnDisk(ref mut data_on_disk) = self._reader {
            Some(ActualReader::InMem(data_on_disk.load_to_memory(self.batch_size)))
        } else {
            None
        };
        if new_reader.is_some() {
            self._reader = new_reader.unwrap();
        }
    }
}


fn fetch_and_reset<T: ExamplesReader>(reader: &mut T, batch_size: usize, reset: bool) -> Vec<Example> {
    let ret = reader.fetch(batch_size);
    if reset {
        reader.reset();
    }
    ret
}