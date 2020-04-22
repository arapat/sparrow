mod gamma;
pub mod model_sync;
mod model_with_version;
pub mod scheduler;

use std::thread::spawn;

use commons::bins::Bins;
use commons::Model;
use config::Config;
use commons::channel::Sender;
use self::gamma::Gamma;
use self::model_sync::ModelSync;


pub fn start_model_manager(
    config: &Config,
    init_tree: &Model,
    bins: &Vec<Bins>,
    next_model_sender: Sender<(Model, String)>,
) {
    debug!("Starting the model sync.");
    let gamma = Gamma::new(config.default_gamma, config.min_gamma);
    let mut model_sync = ModelSync::new(
        init_tree,
        config.num_trees,
        &config.exp_name,
        config.min_ess,
        config.min_grid_size,
        gamma,
        next_model_sender,
        bins,
        config.network.len(),
    );
    let (local_name, network, port) =
        (config.local_name.clone(), config.network.clone(), config.port);
    spawn(move || {
        model_sync.run_with_network(local_name, network, port);
    });
}
