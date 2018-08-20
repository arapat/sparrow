/*!
RustBoost is an implementation of TMSN for boosting.

From a high level, RustBoost consists of three components,
(1) stratified storage,
(2) sampler, and
(3) scanner.
The components communicates via channels, which can be seens as shared FIFO queues.

![System Design](https://www.lucidchart.com/publicSegments/view/23e1b351-c8b8-4cd9-a41f-3698a2b7df42/image.png)
*/
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate log;
#[macro_use] extern crate chan;
extern crate bincode;
extern crate bufstream;
extern crate ordered_float;
extern crate rand;
extern crate rayon;
extern crate serde_json;
extern crate serde_yaml;
extern crate threadpool;
extern crate time;
extern crate rust_tmsn;


/// The class of the weak learner, namely a decision stump.
pub mod tree;

/// The implementation of the AdaBoost algorithm with early stopping rule.
mod boosting;
/// A data loader with two independent caches. Alternatively, we use one
/// of the caches to feed data to the boosting algorithm, and the other
/// to load next sample set.
mod buffer_loader; 
/// A stratified storage structor that organize examples on disk according to their weights.
mod stratified_storage;
/// Common functions and classes.
mod commons;
/// Sampling data from the stratified storage.
mod sampler;
/// The class of the training examples.
mod labeled_data;
// mod validator;

use std::sync::mpsc::channel;
use std::sync::mpsc::sync_channel;

use boosting::Boosting;
use buffer_loader::BufferLoader;
use commons::io::create_bufreader;
use sampler::run_sampler;
use stratified_storage::run_stratified;

// Types
// TODO: use generic types for specifing types
use labeled_data::LabeledData;
pub type TFeature = u8;
pub type TLabel = u8;
pub type Example = LabeledData<TFeature, TLabel>;
pub type ExampleWithScore = (Example, (f32, usize));


/// Configuration of the RustBoost
#[derive(Serialize, Deserialize)]
struct Config {
    pub num_examples: usize,
    pub num_features: usize,
    pub range: std::ops::Range<usize>, 
    pub max_sample_size: usize, 
    pub max_bin_size: usize, 
    pub default_gamma: f32,

    pub num_iterations: usize,
    pub max_trials_before_shrink: u32,

    pub channel_size: usize,
    pub buffer_size: usize,
    pub batch_size: usize,
    pub sampler_num_threads: usize,
    pub sampler_init_grid_size: f32,
    pub sampler_average_window_size: usize,

    pub num_examples_per_block: usize,
    pub disk_buffer_filename: String,
    pub num_assigners: usize,
    pub num_selectors: usize,

    pub network: Vec<String>,
    pub port: u16,
    pub local_name: String,
}


pub fn run_rust_boost(
    config_file: String
) {
    // Load configurations
    let config: Config = serde_yaml::from_reader(
        create_bufreader(&config_file)
    ).unwrap();

    // Strata -> Sampler
    let (s_loaded_examples, r_loaded_examples) = chan::sync(config.channel_size);
    // Sampler -> Strata
    let (s_updated_examples, r_updated_examples) = chan::sync(config.channel_size);
    // Sampler -> BufferLoader
    let (s_sampled_examples, r_sampled_examples) = sync_channel(config.channel_size);
    // Booster -> Sampler
    let (s_next_model, r_next_model) = channel();

    let (counts_table, weights_table) = run_stratified(
        config.num_examples,
        config.num_features,
        config.num_examples_per_block,
        config.disk_buffer_filename.as_ref(),
        config.num_assigners,
        config.num_selectors,
        r_updated_examples,
        s_loaded_examples
    );
    run_sampler(
        config.sampler_num_threads,
        config.sampler_init_grid_size,
        config.sampler_average_window_size,
        r_loaded_examples,
        s_updated_examples,
        s_sampled_examples,
        r_next_model
    );
    let buffer_loader = BufferLoader::new(
        config.buffer_size,
        config.batch_size,
        r_sampled_examples,
        false
    );
    let mut booster = Boosting::new(
        buffer_loader,
        config.range,
        config.max_sample_size,
        config.max_bin_size,
        config.default_gamma
    );
    if config.network.len() > 0 {
        booster.enable_network(config.local_name, &config.network, config.port);
    }
    booster.training(
        config.num_iterations,
        config.max_trials_before_shrink
    );
}