use rand::Rng;
use rand::thread_rng;
use commons::Example;

#[derive(Debug)]
pub struct Examples {
    data: Vec<Example>,
    size: usize,
    cursor: usize
}

impl Examples {
    pub fn new(capacity: usize) -> Examples {
        Examples {
            data: Vec::with_capacity(capacity),
            size: 0,
            cursor: 0
        }
    }

    pub fn append(&mut self, example: &Example) {
        self.data.push(example.clone());
        self.size += 1;
    }

    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
    }

    pub fn fetch(&mut self, batch_size: usize) -> Vec<Example> {
        let curr = self.cursor;
        self.cursor += batch_size;
        self.data[curr..self.cursor].to_vec()
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    pub fn shuffle(&mut self) {
        thread_rng().shuffle(&mut self.data);
    }
}
