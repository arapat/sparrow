use std::collections::HashMap;

use math::round::ceil;
use commons::bins::Bins;


// TODO: extends to more than 2 features for generating the grid
// bounding [ threshold[t-k], threshold[t] ), k is the grid size

pub struct Grid {
    num_splits_on: usize,
    splits_on: Vec<usize>,
    range: Vec<usize>,
    num_workers: usize,
    status: HashMap<String, Option<usize>>,
    grid_size: usize,
}

impl Grid {
    pub fn new(num_splits_on: usize, num_workers: usize, bins: &Vec<Bins>) -> Grid {
        let mut grid = Grid {
            num_splits_on: num_splits_on,
            splits_on: vec![],
            range: bins.iter().map(|t| t.len()).collect(),
            num_workers: num_workers,
            status: HashMap::new(),
            grid_size: 1,
        };
        grid.reset_splits_on();
    }

    pub fn reset_splits_on(&mut self) {
        loop {
            self.splits_on = rand::thread_rng().sample_slice(0..range, count);
            let total_grid = self.splits_on.iter().map(|i| self.range[i].log2()).sum();
            let target_grid = (10 * self.num_workers).log2();
            if total_grid >= target_grid {
                let logsize = (total_grid - target_grid) / self.num_splits_on;
                self.grid_size = (2.0).power(logsize) as usize;
                self.splits_on.sort();
                break;
            }
        }
    }

    pub fn get_new_grid(&mut self, scanner_id: usize) -> (String, Vec<(usize, usize, usize)>) {
        loop {
            let thresholds = self.splits_on.iter().map(|i| {
                let range = self.range[i];
                let discr = (range / self.grid_size).ceil();
                let thr = min((rand::random::<usize>() % discr + 1) * self.grid_size, range)
                (i, thr - self.grid_size, thr)
            }.collect();
            let hashkey = thresholds.iter()
                                    .map(|(index, thr_l, thr_r)| format!("{},{}", index, thr_r))
                                    .join(";");
            if !self.status.contains_key(&key) || self.status[&key].is_none() {
                self.status.insert(key, Some(scanner_id));
                return (key, thresholds);
            }
        }
    }

    pub fn release_grid(&mut self, key: String) {
        self.status.insert(key, None);
    }
}
