pub mod gamma;
pub mod kdtree;
pub mod packet_stats;
pub mod scheduler;

use commons::Model;
use config::Config;
use head::model_manager::model_with_version::ModelWithVersion;
use self::scheduler::Scheduler;


pub fn start_scheduler_async(
    config: &Config,
    num_machines: usize,
    init_tree: &mut Model,
) {
    debug!("Starting the scheduler.");
    let mut model = ModelWithVersion::new(init_tree.clone(), "Sampler".to_string());
    let scheduler = Scheduler::new(
        num_machines,
        &config.exp_name,
        config.min_grid_size,
        &mut model,
        config,
    );
}
