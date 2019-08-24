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
extern crate s3;
extern crate serde_json;
extern crate serde_yaml;
extern crate threadpool;
extern crate time;
extern crate tmsn;
extern crate metricslib;

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
/// Syncing model to S3
mod model_sync;

use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::sync::RwLock;
use std::thread::sleep;
use std::time::Duration;
use tmsn::network::start_network_only_recv;

use bincode::serialize;
use bincode::deserialize;
use booster::Boosting;
use buffer_loader::BufferLoader;
use model_sync::start_model_sync;
use stratified_storage::StratifiedStorage;
use stratified_storage::serial_storage::SerialStorage;
use testing::validate;

use std::sync::mpsc;
use commons::Model;
use commons::bins::create_bins;
use commons::bins::Bins;
use commons::channel;
use commons::io::clear_s3_bucket;
use commons::io::create_bufreader;
use commons::io::create_bufwriter;
use commons::io::load_s3;
use commons::io::raw_read_all;
use commons::io::write_s3;
use commons::performance_monitor::PerformanceMonitor;
use tree::Tree;

// Types
// TODO: decide TFeature according to the bin size
use labeled_data::LabeledData;
pub type RawTFeature = f32;
pub type TFeature = u8;
pub type TLabel = i8;
pub type RawExample = LabeledData<RawTFeature, TLabel>;
pub type Example = LabeledData<TFeature, TLabel>;

pub const FILENAME: &str = "bins.json";
pub const REGION:   &str = "us-east-1";
pub const BUCKET:   &str = "tmsn-cache2";
pub const S3_PATH:  &str = "sparrow-bins/";


/// Configuration for training and testing with Sparrow
#[derive(Serialize, Deserialize)]
pub struct Config {
    /// File path to the training data
    pub training_filename: String,
    /// Number of training examples
    pub num_examples: usize,
    /// Number of features
    pub num_features: usize,
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
    /// Sampling mode: Read/write from memory/local disk/S3
    pub sampling_mode: String,
    /// Worker mode: could be "scanner", "sampler", or "both"
    pub sampler_scanner: String,
    /// Sleep duration: the frequency of loading disk from memory/local disk/S3
    pub sleep_duration: usize,

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
    /// Folder for writing data to S3
    pub exp_name: String,

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

    /// Continous training from an interupted training process
    pub resume_training: bool,
}


fn prep_training(config_file: &String) -> (Config, BufferLoader, Vec<Bins>) {
    // Load configurations
    let config: Config = serde_yaml::from_reader(
        create_bufreader(config_file)
    ).unwrap();

    // Clear S3 before running
    if config.sampling_mode.to_lowercase() == "s3" {
        clear_s3_bucket(REGION, BUCKET, config.exp_name.as_str());
    }

    info!("Creating bins.");
    let s3_path = format!("{}/{}", config.exp_name, S3_PATH);
    let bins = {
        if config.sampler_scanner == "sampler" {
            let mut serial_training_loader = SerialStorage::new(
                config.training_filename.clone(),
                config.num_examples,
                config.num_features,
                true,
                config.positive.clone(),
                None,
            );
            let bins = create_bins(
                config.max_sample_size, config.max_bin_size, config.num_features, &mut serial_training_loader);
            {
                let mut file_buffer = create_bufwriter(&"models/bins.json".to_string());
                let json = serde_json::to_string(&bins).expect("Bins cannot be serialized.");
                file_buffer.write(json.as_ref()).unwrap();
            }
            write_s3(REGION, BUCKET, s3_path.as_str(), FILENAME, &serialize(&bins).unwrap());
            bins
        } else {
            let mut bins = load_s3(REGION, BUCKET, s3_path.as_str(), FILENAME);
            while bins.is_none() {
                bins = load_s3(REGION, BUCKET, s3_path.as_str(), FILENAME);
            }
            let bins = deserialize(&bins.unwrap().0).unwrap();
            {
                let mut file_buffer = create_bufwriter(&"models/bins.json".to_string());
                let json = serde_json::to_string(&bins).expect("Bins cannot be serialized.");
                file_buffer.write(json.as_ref()).unwrap();
            }
            bins
        }
    };
    info!("Starting the buffered loader.");
    let buffer_loader = BufferLoader::new(
        config.buffer_size,
        config.batch_size,
        config.channel_size,
        config.sampling_mode.clone(),
        config.sleep_duration,
        false, // config.sampler_scanner != "sampler",
        Some(config.min_ess),
        config.sampler_scanner.clone(),
        config.exp_name.clone(),
    );
    (config, buffer_loader, bins)
}


pub fn training(config_file: String) {
    let mut training_perf_mon = PerformanceMonitor::new();
    training_perf_mon.start();

    let (config, buffer_loader, bins) = prep_training(&config_file);
    // Resuming from an earlier training:
    //     The program needs three files to resume from an earlier training:
    //         1. model file, `model.json`, which is being loaded below;
    //         2. last sample, `lastest_sample.bin`, which is sent out to scanner as the first
    //            sample;
    //         3. strata snapshot, `stratified.serde`, which will be loaded during the
    //            initialization of the `stratified_structure` object below.
    let init_tree = {
        if config.resume_training {
            let (_, _, model): (f32, usize, Model) =
                serde_json::from_str(&raw_read_all(&"model.json".to_string()))
                        .expect(&format!("Cannot parse the model in `model.json`"));
            model
        } else {
            Tree::new(config.num_iterations, 0.0, 0.0)
        }
    };
    if config.sampler_scanner == "scanner" {
        info!("Starting the booster.");
        let mut booster = Boosting::new(
            config.exp_name.clone(),
            init_tree.clone(),
            config.num_iterations,
            config.num_features,
            config.min_gamma,
            buffer_loader,
            bins,
            config.max_sample_size,
            config.default_gamma,
            config.save_process,
            config.save_interval,
        );
        booster.enable_network(config.local_name, config.port);
        booster.set_root_tree();
        booster.training(training_perf_mon.get_duration());
    } else { // if config.sampler_scanner == "sampler" {
        let sampler_state = Arc::new(RwLock::new(true));
        info!("Starting the model sync.");
        // Pass the models between the network to the Strata
        let (next_model_s, next_model_r) = channel::bounded(config.channel_size, "updated-models");
        start_model_sync(
            init_tree.clone(), config.local_name.clone(), config.num_iterations,
            config.network.clone(), config.port, next_model_s,
            config.default_gamma, config.min_gamma,
            buffer_loader.current_sample_version.clone(), config.exp_name.clone(),
            sampler_state.clone());
        info!("Starting the stratified structure.");
        let stratified_structure = StratifiedStorage::new(
            init_tree,
            config.num_examples,
            config.num_features,
            config.positive.clone(),
            config.num_examples_per_block,
            config.disk_buffer_filename.as_ref(),
            config.num_assigners,
            config.num_samplers,
            buffer_loader.sampled_examples_s.clone(),
            next_model_r,
            config.channel_size,
            sampler_state.clone(),
            config.debug_mode,
            config.resume_training,
        );
        if !config.resume_training {
            info!("Initializing the stratified structure.");
            stratified_structure.init_stratified_from_file(
                config.training_filename.clone(),
                config.num_examples,
                config.batch_size,
                config.num_features,
                bins.clone(),
            );
        }
        let (hb_s, hb_r): (mpsc::Sender<String>, mpsc::Receiver<String>) =
            mpsc::channel();
        start_network_only_recv(config.local_name.as_ref(), &config.network, config.port + 1, hb_s);
        let mut state = true;
        while state {
            // Check if termination is manually requested
            let filename = "status.txt".to_string();
            if Path::new(&filename).exists() && raw_read_all(&filename).trim() == "0".to_string() {
                info!("Change in the status.txt has been detected.");
                *(sampler_state.write().unwrap()) = false;
            }
            // Check if any one of the scanners is still working
            let mut hb_count = 0;
            while hb_r.try_recv().is_ok() {
                hb_count += 1;
            }
            if hb_count == 0 {
                info!("All scanners are dead.");
                *(sampler_state.write().unwrap()) = false;
            }
            state = {
                let t = sampler_state.read().unwrap();
                *t
            };
            sleep(Duration::from_secs(20));
        }
        info!("State has been set to false. Main process to exit in 120 seconds.");
        sleep(Duration::from_secs(120));
    }
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
