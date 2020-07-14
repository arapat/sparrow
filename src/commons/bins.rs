use ordered_float::NotNaN;

use std::collections::BTreeMap;

use TFeature;
use config::Config;
use commons::persistent_io::read_bins_disk;
use commons::persistent_io::read_bins_s3;
use commons::persistent_io::write_bins_disk;
use commons::persistent_io::write_bins_s3;
use head::sampler::stratified_storage::serial_storage::SerialStorage;


// If discretized to 1 value, keep the largest value;
// if discretized to 2 values, keep the median and the largest values;
// etc.

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
        let avg_bin_size = (distinct_vals.total_vals as f32 / size as f32).ceil() as usize;
        let mut last_val = 0.0;
        let mut counter = 0;
        let mut vals: Vec<f32> = vec![];
        distinct_vals.distinct.iter().for_each(|(k, v)| {
            counter += *v as usize;
            let k = k.into_inner();
            if counter > avg_bin_size {
                vals.push(k);
                counter = 0;
            }
            last_val = k;
        });
        if counter > 0 {
            vals.push(last_val);
        }
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
        let mut left = 0;
        let mut right = self.size;
        while left + 1 < right {
            let medium = (left + right) / 2;
            if self.vals[medium] <= val {
                left = medium;
            } else {
                right = medium;
            }
        }
        left as TFeature
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


/// Create bins for the features
///
/// * `max_sample_size`: The number of examples to read for deciding the splitting
///   thresholds for creating bins
/// * `max_bin_size`: The total number of bins created for each freature. The actual
///   number of bins might be smaller if there are fewer distinct values for a feature
/// * `data_loader`: Data loader for providing training examples
fn create_bins(
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
    let bins: Vec<Bins> = distinct.iter()
                                 .map(|mapper| Bins::new(max_bin_size, mapper))
                                 .collect();
    // Logging
    let total_bins: usize = bins.iter().map(|t| t.len()).sum();
    info!("Bins are created. {} Features. {} Bins.", bins.len(), total_bins);
    bins
}

pub fn load_bins(mode: &str, config: Option<&Config>) -> Vec<Bins> {
    if mode == "sampler" {
        let config = config.unwrap();
        let mut serial_training_loader = SerialStorage::new(
            config.training_filename.clone(),
            config.num_examples,
            config.num_features,
            true,
            config.positive.clone(),
            None,
        );
        let bins = create_bins(
            config.max_sample_size, config.max_bin_size, config.num_features,
            &mut serial_training_loader);
        write_bins_disk(&bins);
        write_bins_s3(&bins, &config.exp_name);
        bins
    } else if mode == "scanner" {
        let config = config.unwrap();
        read_bins_s3(&config.exp_name)
    } else {  // if mode == "testing"
        read_bins_disk()
    }
}