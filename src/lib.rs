/*!
RustBoost is an implementation of TMSN for boosting.

From a high level, RustBoost consists of three components,
(1) stratified storage,
(2) sampler, and
(3) scanner.
The components communicates via channels, which can be seens as shared FIFO queues.

![System Design](https://www.lucidchart.com/publicSegments/view/23e1b351-c8b8-4cd9-a41f-3698a2b7df42/image.png)
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


/// The class of the weak learner, namely a decision stump.
pub mod tree;

/// The implementation of the AdaBoost algorithm with early stopping rule.
mod booster;
/// A data loader with two independent caches. Alternatively, we use one
/// of the caches to feed data to the boosting algorithm, and the other
/// to load next sample set.
mod buffer_loader; 
/// A stratified storage structor that organize examples on disk according to their weights.
mod stratified_storage;
/// Common functions and classes.
mod commons;
/// The class of the training examples.
mod labeled_data;
/// Validating models

use booster::Boosting;
use buffer_loader::BufferLoader;
use stratified_storage::StratifiedStorage;
use stratified_storage::serial_storage::SerialStorage;

use commons::channel;
use commons::io::create_bufreader;

// Types
// TODO: use generic types for specifing types
use labeled_data::LabeledData;
pub type TFeature = f32;
pub type TLabel = u8;
pub type Example = LabeledData<TFeature, TLabel>;


/// Configuration of the RustBoost
#[derive(Serialize, Deserialize)]
struct Config {
    pub training_filename: String,
    pub training_is_binary: bool,
    pub training_bytes_per_example: usize,

    pub testing_filename: String,
    pub testing_is_binary: bool,
    pub testing_bytes_per_example: usize,

    pub num_examples: usize,
    pub num_testing_examples: usize,
    pub num_features: usize,
    pub range: std::ops::Range<usize>, 
    pub max_sample_size: usize, 
    pub max_bin_size: usize, 
    pub min_gamma: f32,
    pub default_gamma: f32,
    pub min_ess: f32,

    pub num_iterations: usize,
    pub max_leaves: usize,
    pub max_trials_before_shrink: u32,

    pub channel_size: usize,
    pub buffer_size: usize,
    pub batch_size: usize,
    pub serial_sampling: bool,

    pub num_examples_per_block: usize,
    pub disk_buffer_filename: String,
    pub num_assigners: usize,
    pub num_samplers: usize,

    pub network: Vec<String>,
    pub port: u16,
    pub local_name: String,
    pub debug_mode: bool,
}


pub fn run_rust_boost(config_file: String) {
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

    info!("Starting the stratified structure.");
    let stratified_structure = StratifiedStorage::new(
        config.num_examples,
        config.num_features,
        config.num_examples_per_block,
        config.disk_buffer_filename.as_ref(),
        config.num_assigners,
        config.num_samplers,
        sampled_examples_s,
        sampling_signal_r,
        next_model_r,
        config.channel_size,
    );
    info!("Initializing the stratified structure.");
    stratified_structure.init_stratified_from_file(
        config.training_filename.clone(),
        config.num_examples,
        config.batch_size,
        config.num_features,
        config.training_is_binary,
        Some(config.training_bytes_per_example),
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
    let serial_training_loader = SerialStorage::new(
        config.training_filename.clone(),
        config.num_examples,
        config.num_features,
        false,
        None,
        false,
    );
    info!("Starting the booster.");
    let mut booster = Boosting::new(
        config.num_iterations,
        config.max_leaves,
        config.min_gamma,
        config.max_trials_before_shrink,
        buffer_loader,
        serial_training_loader,
        config.range,
        config.max_sample_size,
        config.max_bin_size,
        config.default_gamma,
        next_model_s,
        config.debug_mode,
    );
    if config.network.len() > 0 {
        booster.enable_network(config.local_name, &config.network, config.port);
    }
    booster.training();
}
