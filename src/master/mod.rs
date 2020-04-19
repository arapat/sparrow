
pub mod model_manager;
pub mod sampler;

use config::Config;
use config::SampleMode;
use commons::bins::Bins;
use commons::Model;

use commons::channel;
use self::sampler::start_sampler;
use self::model_manager::start_model_manager;

pub fn start_master(
    config: &Config,
    sample_mode: &SampleMode,
    bins: &Vec<Bins>,
    init_tree: &Model,
) {
    // Pass the models between the network to the Strata
    let (next_model_s, next_model_r) = channel::bounded(config.channel_size, "updated-models");

    start_sampler(
        config,
        sample_mode,
        bins,
        init_tree,
        next_model_r,
    );
    start_model_manager(
        config,
        init_tree,
        bins,
        next_model_s,
    );
}
