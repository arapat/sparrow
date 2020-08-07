pub mod model_sync;
pub mod model_with_version;

use std::thread::spawn;

use commons::bins::Bins;
use commons::Model;
use config::Config;
use commons::channel::Sender;
use self::model_sync::ModelSync;


pub fn start_model_manager_async(
    config: &Config,
    init_tree: &Model,
    bins: &Vec<Bins>,
    next_model_sender: Sender<(Model, String)>,
) {
    debug!("Starting the model sync.");
    let mut model_sync = ModelSync::new(
        init_tree,
        config.num_trees,
        &config.exp_name,
        config.min_ess,
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
