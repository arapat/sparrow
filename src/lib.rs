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

/// Common functions and classes.
pub mod commons;
mod config;
/// Validating models
mod testing;
mod master;
mod scanner;

use config::Config;
use config::SampleMode;
use commons::Model;
use commons::bins::Bins;
use commons::tree::ADTree as Tree;

use scanner::start as start_scanner;
use master::start_master;
use testing::validate;

use commons::bins::load_bins;
use commons::io::clear_s3_bucket;
use commons::persistent_io::read_model;

// Types
// TODO: decide TFeature according to the bin size
use commons::labeled_data::LabeledData;
pub type RawTFeature = f32;
pub type TFeature = u8;
pub type TLabel = i8;
pub type RawExample = LabeledData<RawTFeature, TLabel>;
pub type Example = LabeledData<TFeature, TLabel>;

pub const REGION:   &str = "us-east-1";
pub const BUCKET:   &str = "tmsn-cache2";


fn prep_training(config_filepath: &String) -> (Config, SampleMode, Vec<Bins>) {
    // Load configurations
    let config = Config::new(config_filepath);
    let sample_mode = SampleMode::new(&config.sampling_mode);

    // Clear S3 before running
    if sample_mode == SampleMode::S3 {
        clear_s3_bucket(REGION, BUCKET, config.exp_name.as_str());
    }

    debug!("Loading bins.");
    let bins = load_bins(config.sampler_scanner.as_str(), Some(&config));
    (config, sample_mode, bins)
}


pub fn training(config_filepath: &String) {
    let (config, sample_mode, bins) = prep_training(config_filepath);
    let init_tree: Model = {
        if config.resume_training && config.sampler_scanner == "sampler" {
            // Resuming from an earlier training
            debug!("resume_training is enabled");
            let (_, _, mut model) = read_model();
            model.base_version = 0;
            debug!("Loaded an existing tree");
            model
        } else {
            debug!("Created a new tree");
            // TODO: extend for the cases that more than 4 nodes were used for creating grids
            Tree::new(config.num_trees * (4 + config.num_splits + 1) + 10)
        }
    };
    if config.sampler_scanner == "scanner" {
        start_scanner(&config, &sample_mode, &bins, &init_tree);
    } else { // if config.sampler_scanner == "sampler"
        start_master(&config, &sample_mode, &bins, &init_tree);
    }
}


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
