/*!
Sparrow is an implementation of TMSN for boosting.

From a high level, Sparrow consists of three components,

1. scanner: it runs the boosting process, which scans the samples in memory, and
updates the current model by finding a new weak rule to be added to the score function;
2. sampler: it samples examples from disk and updates their scores according to the latest
score function,
3. model manager: it assigns tasks to the scanners, receives model updates from them, and
maintains the current score function.
*/
#[macro_use] extern crate crossbeam_channel;
#[macro_use] extern crate lazy_static;
#[macro_use] extern crate log;
#[macro_use] extern crate serde_derive;
extern crate awscreds;
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

/// Common functions and classes.
mod commons;
mod config;
/// Validating models
mod testing;
/// Implementation of the components running on head node, specifically the scanner
/// and the model manager
pub mod head;
/// Implementation of the scanner
/// ![](/images/scanner.png)
pub mod scanner;

use config::Config;
use config::SampleMode;
use commons::model::Model;
use commons::bins::Bins;

use scanner::start_scanner;
use scanner::handle_network_send;
use head::start_head;
use testing::validate;

use commons::bins::load_bins;
use commons::io::clear_s3_bucket;
use commons::persistent_io::read_model;

// Types
// TODO: decide TFeature according to the bin size
use commons::labeled_data::LabeledData;
type RawTFeature = f32;
type TFeature = u8;
type TLabel = i8;
type RawExample = LabeledData<RawTFeature, TLabel>;
type Example = LabeledData<TFeature, TLabel>;

const REGION:   &str = "us-east-1";
const BUCKET:   &str = "tmsn-cache2";


fn prep_training(config_filepath: &String) -> (Config, SampleMode, Vec<Bins>) {
    // Load configurations
    let config = Config::new(config_filepath);
    let sample_mode = SampleMode::new(&config.sampling_mode);

    // Clear S3 before running
    if sample_mode == SampleMode::S3 {
        clear_s3_bucket(REGION, BUCKET, config.exp_name.as_str());
    }

    debug!("Loading bins.");
    let bins =
        if config.debug_mode {
            load_bins("testing", Some(&config))
        } else {
            load_bins(config.sampler_scanner.as_str(), Some(&config))
        };
    (config, sample_mode, bins)
}

/// Train a model
///
/// Parameter:
///
/// * config_filepath: the filepath to the configuration file
pub fn training(config_filepath: &String) {
    let (config, sample_mode, bins) = prep_training(config_filepath);
    if config.sampler_scanner == "scanner" {
        let (mut network, new_updates_receiver) = start_scanner(config, sample_mode, bins);
        handle_network_send(&mut network, new_updates_receiver);
    } else { // if config.sampler_scanner == "sampler"
        let init_tree: Model = {
            if config.resume_training && config.sampler_scanner == "sampler" {
                // Resuming from an earlier training
                debug!("resume_training is enabled");
                let (_, _, mut model) = read_model();
                model.base_size = 0;
                debug!("Loaded an existing model");
                model
            } else {
                debug!("Created a new model");
                Model::new()
            }
        };
        start_head(config, sample_mode, bins, init_tree);
    }
}


/// Test a model
///
/// Parameter:
///
/// * config_filepath: the filepath to the configuration file
pub fn testing(config_filepath: &String) {
    // Load configurations
    let config: Config = Config::new(config_filepath);
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
