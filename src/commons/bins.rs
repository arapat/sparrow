use ordered_float::NotNaN;

use std::collections::BTreeMap;

use stratified_storage::serial_storage::SerialStorage;
use super::super::TFeature;


// TODO: support NaN feature values
/// The percentiles of a specific feature dimension,
/// which would be used as the candidates weak rules on that dimension.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Bins {
    size: usize,
    vals: Vec<f32>
}

struct DistinctValues {
    total_vals: usize,
    distinct: BTreeMap<NotNaN<f32>, u32>
}

impl Bins {
    fn new(size: usize, distinct_vals: &DistinctValues) -> Bins {
        let avg_bin_size = (distinct_vals.total_vals / size) as usize;
        let mut last_val = 0.0;
        let mut counter = 0;
        let mut vals: Vec<f32> = vec![];
        distinct_vals.distinct.iter().for_each(|(k, v)| {
            let k = k.into_inner();
            if counter > avg_bin_size {
                vals.push((last_val + k) / 2.0);
                counter = 0;
            }
            counter += *v as usize;
            last_val = k;
        });
        Bins {
            size: vals.len(),
            vals: vals
        }
    }

    /// Return the number of thresholds. 
    pub fn len(&self) -> usize {
        self.size
    }

    /// Return the vector of thresholds.
    #[allow(dead_code)]
    pub fn get_vals(&self) -> &Vec<f32> {
        &self.vals
    }

    pub fn get_split_index(&self, val: f32) -> TFeature {
        if self.vals.len() <= 0 || val <= self.vals[0] {
            return 0;
        }
        let mut left = 0;
        let mut right = self.size;
        while left + 1 < right {
            let medium = (left + right) / 2;
            if val <= self.vals[medium] {
                right = medium;
            } else {
                left = medium;
            }
        }
        right as TFeature
    }
}

impl DistinctValues {
    fn new() -> DistinctValues {
        DistinctValues {
            total_vals: 0,
            distinct: BTreeMap::new()
        }
    }

    fn update(&mut self, val: f32) {
        self.total_vals += 1;
        let count = self.distinct.entry(NotNaN::new(val).unwrap()).or_insert(0);
        *count += 1;
    }
}


/// Create bins for the features in the range specified by `range`
///
/// * `max_sample_size`: The number of examples to read for deciding the splitting
///   thresholds for creating bins
/// * `max_bin_size`: The total number of bins created for each freature. The actual
///   number of bins might be smaller if there are fewer distinct values for a feature
/// * `range`: The range of the features this worker is responsible for
/// * `data_loader`: Data loader for providing training examples
pub fn create_bins(
    max_sample_size: usize,
    max_bin_size: usize,
    num_features: usize,
    data_loader: &mut SerialStorage,
) -> Vec<Bins> {
    let mut distinct: Vec<DistinctValues> = Vec::with_capacity(num_features);
    let mut remaining_reads = max_sample_size;

    for _ in 0..num_features {
        distinct.push(DistinctValues::new());
    }
    while remaining_reads > 0 {
        let data = data_loader.read_raw(1000);
        data.iter().for_each(|example| {
            let feature = &(example.feature);
            distinct.iter_mut()
                    .enumerate()
                    .for_each(|(idx, mapper)| {
                        mapper.update(feature[idx] as f32);
                    });
        });
        remaining_reads -= data.len();
    }
    let ret: Vec<Bins> = distinct.iter()
                                 .map(|mapper| Bins::new(max_bin_size, mapper))
                                 .collect();

    // Logging
    let total_bins: usize = ret.iter().map(|t| t.len()).sum();
    info!("Bins are created. {} Features. {} Bins.", ret.len(), total_bins);
    ret
}
