/// Implementation of the model manager
/// ![](/images/taskmanager.png)
pub mod model_manager;
/// Implementation of the sampler
/// ![](/images/sampler.png)
pub mod sampler;

use std::path::Path;
use std::thread::sleep;
use std::time::Duration;

use config::Config;
use config::SampleMode;
use commons::bins::Bins;
use commons::Model;

use commons::channel;
use commons::io::raw_read_all;
use self::sampler::start_sampler_async;
use self::model_manager::start_model_manager_async;

/// Start the components resided in head node
pub fn start_head(
    config: Config,
    sample_mode: SampleMode,
    bins: Vec<Bins>,
    init_tree: Model,
) {
    // Pass the models between the network to the Strata
    let (next_model_s, next_model_r) = channel::bounded(config.channel_size, "updated-models");

    let sampler_state = start_sampler_async(
        &config,
        &sample_mode,
        &bins,
        &init_tree,
        next_model_r,
    );
    start_model_manager_async(
        &config,
        &init_tree,
        &bins,
        next_model_s,
    );

    // Monitor running state, exit if state is false
    loop {
        // Check if termination is manually requested
        let filename = "status.txt".to_string();
        if Path::new(&filename).exists() && raw_read_all(&filename).trim() == "0".to_string() {
            debug!("sampler state, false, change in the status.txt has been detected");
            *(sampler_state.write().unwrap()) = false;
            break;
        }
        sleep(Duration::from_secs(20));
    }
    debug!("State has been set to false. Main process to exit in 120 seconds.");
    sleep(Duration::from_secs(120));
    if std::fs::remove_file("status.txt").is_ok() {
        debug!("removed `status.txt`");
    }
}
