use rand::Rng;
use rand::thread_rng;
use commons::Example;

use super::ExamplesReader;


#[derive(Debug)]
pub struct DataInMem {
    data: Vec<Example>,
    size: usize,
    cursor: usize
}


impl ExamplesReader for DataInMem {
    fn fetch(&mut self, batch_size: usize) -> Vec<Example> {
        let curr = self.cursor;
        self.cursor += batch_size;
        self.data[curr..self.cursor].to_vec()
    }

    fn reset(&mut self) {
        self.cursor = 0;
    }
}


impl DataInMem {
    pub fn new(capacity: usize) -> DataInMem {
        DataInMem {
            data: Vec::with_capacity(capacity),
            size: 0,
            cursor: 0
        }
    }

    pub fn append(&mut self, example: &Example) {
        self.data.push(example.clone());
        self.size += 1;
    }

    pub fn release(&mut self, shuffle: bool) {
        self.data.shrink_to_fit();
        if shuffle {
            thread_rng().shuffle(&mut self.data);
        }
    }
}