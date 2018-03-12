extern crate ordered_float;

use std::collections::BTreeMap;
use self::ordered_float::NotNaN;

use data_loader::DataLoader;


// TODO: support NaN feature values
pub struct Bins {
    size: usize,
    vals: Vec<f32>
}

struct BinMapper {
    total_vals: usize,
    distinct: BTreeMap<NotNaN<f32>, u32>
}

impl Bins {
    fn new(size: usize, mapper: &BinMapper) -> Bins {
        let avg_bin_size = (mapper.total_vals / size) as usize;
        let mut last_val = 0.0;
        let mut counter = 0;
        let mut vals: Vec<f32> = vec![];
        mapper.distinct.iter().for_each(|(wrapped_k, v)| {
            let k = wrapped_k.into_inner();
            if counter > avg_bin_size {
                vals.push((last_val + k) / 2.0);
                counter = 0;
            }
            counter += *v as usize;
            last_val = k;
        });
        let actual_size = vals.len();
        Bins {
            size: actual_size,
            vals: vals
        }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn get_vals(&self) -> &Vec<f32> {
        &self.vals
    }
}

impl BinMapper {
    fn new() -> BinMapper {
        BinMapper {
            total_vals: 0,
            distinct: BTreeMap::new()
        }
    }

    fn update(&mut self, val: &f32) {
        self.total_vals += 1;
        let count = self.distinct.entry(NotNaN::new(*val).unwrap()).or_insert(0);
        *count += 1;
    }
}


pub fn create_bins(max_sample_size: usize, max_bin_size: usize, data_loader: &mut DataLoader) -> Vec<Bins> {
    let feature_size = data_loader.get_feature_size();
    let mut mappers: Vec<BinMapper> = Vec::with_capacity(feature_size);
    let mut remaining_reads = max_sample_size;

    for _ in 0..feature_size {
        mappers.push(BinMapper::new());
    }
    while remaining_reads > 0 {
        data_loader.fetch_next_batch();
        let data = data_loader.get_curr_batch();
        data.iter().for_each(|example| {
            let features = example.get_features();
            mappers.iter_mut()
                   .zip(0..feature_size)
                   .for_each(|(mapper, idx)| {
                       mapper.update(&features[idx]);
                   });
        });
        remaining_reads -= data.len();
    }
    mappers.iter()
           .map(|mapper| Bins::new(max_bin_size, mapper))
           .collect()
}
