use std::collections::HashMap;

use commons::bins::Bins;


type KeyType = (usize, usize, usize, usize);

struct Grid {
    range: Vec<usize>,
    status: HashMap<(usize, usize, usize, usize), Option<usize>>,
}

impl Grid {
    pub fn new(bins: &Vec<Bins>) -> Grid {
        Grid {
            range: bins.iter().map(|t| t.len() + 1).collect(),
            status: HashMap::new(),
        }
    }

    pub fn get_new_grid(&mut self, scanner_id: usize) -> KeyType {
        loop {
            let f1   = rand::random::<usize>() % self.range.len();
            let thr1 = rand::random::<usize>() % self.range[f1];
            let f2 =   rand::random::<usize>() % self.range.len();
            let thr2 = rand::random::<usize>() % self.range[f2];
            let key = (f1, thr1, f2, thr2);
            if !self.status.contains_key(&key) || self.status[&key].is_none() {
                self.status.insert(key, Some(scanner_id));
                return key;
            }
        }
    }

    pub fn release_grid(&mut self, key: KeyType) {
        self.status.insert(key, None);
    }
}
