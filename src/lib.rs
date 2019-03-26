/*!
Sparrow is an implementation of TMSN for boosting.

From a high level, Sparrow consists of three components,
(1) stratified storage,
(2) sampler, and
(3) scanner.
The components communicates via channels, which can be seens as shared FIFO queues.

![System Design](https://www.lucidchart.com/publicSegments/view/b6ebfe77-33fe-4937-94e2-8a91175e355f/image.png)
*/
#[macro_use] extern crate crossbeam_channel;
#[macro_use] extern crate lazy_static;
#[macro_use] extern crate log;
#[macro_use] extern crate serde_derive;
extern crate bincode;
extern crate bufstream;
extern crate evmap;
extern crate ordered_float;
extern crate rand;
extern crate rayon;
extern crate serde_json;
extern crate serde_yaml;
extern crate threadpool;
extern crate time;
extern crate tmsn;
extern crate metricslib;

use std::io::Write;

/// The class of the weak learner, namely a decision stump.
mod tree;

/// The implementation of the AdaBoost algorithm with early stopping rule.
mod booster;
/// A data loader with two independent caches. Alternatively, we use one
/// of the caches to feed data to the boosting algorithm, and the other
/// to load next sample set.
mod buffer_loader; 
/// Common functions and classes.
mod commons;
/// The class of the training examples.
mod labeled_data;
/// A stratified storage structor that organize examples on disk according to their weights.
mod stratified_storage;
/// Validating models
mod testing;

use booster::Boosting;
use buffer_loader::BufferLoader;
use stratified_storage::StratifiedStorage;
use stratified_storage::serial_storage::SerialStorage;
use testing::validate;

use commons::bins::create_bins;
use commons::channel;
use commons::io::create_bufreader;
use commons::io::create_bufwriter;
use commons::performance_monitor::PerformanceMonitor;

// Types
// TODO: decide TFeature according to the bin size
use labeled_data::LabeledData;
pub type RawTFeature = f32;
pub type TFeature = u8;
pub type TLabel = i8;
pub type RawExample = LabeledData<RawTFeature, TLabel>;
pub type Example = LabeledData<TFeature, TLabel>;


/// Configuration for training and testing with Sparrow
#[derive(Serialize, Deserialize)]
pub struct Config {
    /// File path to the training data
    pub training_filename: String,
    /// Number of training examples
    pub num_examples: usize,
    /// Number of features
    pub num_features: usize,
    /// Range of the features for creating weak rules
    pub range: std::ops::Range<usize>, 
    /// Label for positive examples
    pub positive: String,
    /// File path to the testing data
    pub testing_filename: String,
    /// Number of testing examples
    pub num_testing_examples: usize,

    /// Number of examples to scan for generating heuristic used in Sparrow
    pub max_sample_size: usize, 
    /// Maximum number of bins for discretizing continous feature values
    pub max_bin_size: usize, 
    /// Minimum value of the \gamma of the generated tree nodes
    pub min_gamma: f32,
    /// Default maximum value of the \gamma for generating tree nodes
    pub default_gamma: f32,
    /// Maximum number of examples to scan before shrinking the value of \gamma
    pub max_trials_before_shrink: u32,
    /// Minimum effective sample size for triggering resample
    pub min_ess: f32,

    /// Number of boosting iterations
    pub num_iterations: usize,
    /// Maximum number of tree leaves in each boosted tree
    pub max_leaves: usize,

    /// Maximum number of elements in the channel connecting scanner and sampler
    pub channel_size: usize,
    /// Number of examples in the sample set that needs to be loaded into memory
    pub buffer_size: usize,
    /// Number of examples to process in each weak rule updates
    pub batch_size: usize,
    /// Set to true to stop running sampler in the background of the scanner
    pub serial_sampling: bool,

    /// Number of examples in a block on the stratified binary file
    pub num_examples_per_block: usize,
    /// File name for the stratified binary file
    pub disk_buffer_filename: String,
    /// Number of threads for putting examples back to correct strata
    pub num_assigners: usize,
    /// Number of threads for sampling examples from strata
    pub num_samplers: usize,

    /// IP addresses of other machines in the network
    pub network: Vec<String>,
    /// The network port used for parallel training
    pub port: u16,
    /// Identifier for the local machine
    pub local_name: String,

    /// Flag for keeping all intermediate models during training (for debugging purpose)
    pub save_process: bool,
    /// Number of iterations between persisting models on disk
    pub save_interval: usize,
    /// Flag for activating debug mode
    pub debug_mode: bool,

    /// (for validation only) the file names of the models to run the validation
    pub models_table_filename: String,
    /// Flag indicating if models are trained incrementally
    pub incremental_testing: bool,
    /// Flag for validation mode, set to true to output raw scores of testing examples,
    /// and set to false for printing the validation scores but not raw scores
    pub testing_scores_only: bool,
}


pub fn training(config_file: String) {
    let mut training_perf_mon = PerformanceMonitor::new();
    training_perf_mon.start();

    // Load configurations
    let config: Config = serde_yaml::from_reader(
        create_bufreader(&config_file)
    ).unwrap();

    // Strata -> BufferLoader
    let (sampled_examples_s, sampled_examples_r) = channel::bounded(config.channel_size, "gather-samples");
    // BufferLoader -> Strata
    let (sampling_signal_s, sampling_signal_r) = channel::bounded(10, "sampling-signal");
    // Booster -> Strata
    let (next_model_s, next_model_r) = channel::bounded(config.channel_size, "updated-models");

    info!("Creating bins.");
    let mut serial_training_loader = SerialStorage::new(
        config.training_filename.clone(),
        config.num_examples,
        config.num_features,
        true,
        config.positive.clone(),
        None,
        config.range.clone(),
    );
    let bins = create_bins(
        config.max_sample_size, config.max_bin_size, &config.range, &mut serial_training_loader);
    {
        let mut file_buffer = create_bufwriter(&"models/bins.json".to_string());
        let json = serde_json::to_string(&bins).expect("Bins cannot be serialized.");
        file_buffer.write(json.as_ref()).unwrap();
    }
    let validate_set1: Vec<Example> = {
        if false {
            let mut loader = SerialStorage::new(
                config.testing_filename.clone(),
                config.num_testing_examples,
                config.num_features,
                true,
                config.positive.clone(),
                Some(bins.clone()),
                config.range.clone(),
            );
            let mut ret = Vec::with_capacity(config.num_testing_examples);
            while ret.len() < config.num_testing_examples {
                ret.extend(loader.read(config.batch_size));
            }
            ret
        } else {
            vec![]
        }
    };
    let validate_set2: Vec<Example> = {
        if false {
            let mut loader = SerialStorage::new(
                config.training_filename.clone(),
                config.num_examples,
                config.num_features,
                true,
                config.positive.clone(),
                Some(bins.clone()),
                config.range.clone(),
            );
            let mut ret = Vec::with_capacity(config.num_examples);
            while ret.len() < config.num_examples {
                ret.extend(loader.read(config.batch_size));
            }
            ret
        } else {
            vec![]
        }
    };
    info!("Starting the stratified structure.");
    let stratified_structure = StratifiedStorage::new(
        config.num_examples,
        config.num_features,
        config.positive.clone(),
        config.num_examples_per_block,
        config.disk_buffer_filename.as_ref(),
        config.num_assigners,
        config.num_samplers,
        sampled_examples_s,
        sampling_signal_r,
        next_model_r,
        config.channel_size,
        config.debug_mode,
    );
    info!("Initializing the stratified structure.");
    stratified_structure.init_stratified_from_file(
        config.training_filename.clone(),
        config.num_examples,
        config.batch_size,
        config.num_features,
        config.range.clone(),
        bins.clone(),
    );
    info!("Starting the buffered loader.");
    let buffer_loader = BufferLoader::new(
        config.buffer_size,
        config.batch_size,
        sampled_examples_r,
        sampling_signal_s,
        config.serial_sampling,
        true,
        Some(config.min_ess),
    );
    info!("Starting the booster.");
    let mut booster = Boosting::new(
        config.num_iterations,
        config.min_gamma,
        config.max_trials_before_shrink,
        buffer_loader,
        // serial_training_loader,
        bins,
        config.range,
        config.max_sample_size,
        config.default_gamma,
        next_model_s,
        config.save_process,
        config.save_interval,
    );
    if config.network.len() > 0 {
        booster.enable_network(config.local_name, &config.network, config.port);
    }
    booster.training(training_perf_mon.get_duration(), validate_set1, validate_set2);
}


pub fn testing(config_file: String) {
    // Load configurations
    let config: Config = serde_yaml::from_reader(
        create_bufreader(&config_file)
    ).unwrap();
    validate(
        config.models_table_filename.clone(),
        config.testing_filename.clone(),
        config.num_testing_examples,
        config.num_features,
        config.batch_size,
        config.positive.clone(),
        config.incremental_testing,
        config.testing_scores_only,
    );
}