use ordered_float::NotNaN;

use std::collections::BTreeMap;
use std::ops::Range;

use buffer_loader::BufferLoader;


// TODO: support NaN feature values
/// The percentiles of a specific feature dimension,
/// which would be used as the candidates weak rules on that dimension.
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

    /// Return the number of thresholds. 
    pub fn len(&self) -> usize {
        self.size
    }

    /// Return the vector of thresholds.
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


/// Find the percentiles of the feature in the range specified by `range`. 
///
/// To decide the percentiles, the function reads `max_sample_size` number 
/// of examples from `data_loader`.
/// The number of selected percentiles is `max_bin_size`.
///
/// If the number of different values on this feature dimension `k` is smaller
/// than `max_bin_size`, only `k` percentiles would return.
pub fn create_bins(max_sample_size: usize, max_bin_size: usize,
                   range: &Range<usize>, data_loader: &mut BufferLoader) -> Vec<Bins> {
    let start = range.start;
    let range_size = range.end - start;
    let mut mappers: Vec<BinMapper> = Vec::with_capacity(range_size);
    let mut remaining_reads = max_sample_size;

    for _ in 0..range_size {
        mappers.push(BinMapper::new());
    }
    while remaining_reads > 0 {
        let data = data_loader.get_next_batch(false);
        data.iter().for_each(|example| {
            let features = example.0.get_features();
            mappers.iter_mut()
                   .enumerate()
                   .for_each(|(idx, mapper)| {
                       mapper.update(&(features[idx + start] as f32));
                   });
        });
        remaining_reads -= data.len();
    }
    let ret: Vec<Bins> = mappers.iter()
                                .map(|mapper| Bins::new(max_bin_size, mapper))
                                .collect();

    info!("Bins are created.");
    ret.iter()
       .enumerate()
       .for_each(|(idx, bin)| {
           info!("{} thresholds are generated for the feature {}.", bin.len(), idx + start);
       });
    ret
}
