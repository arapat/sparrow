/// The implementation of the AdaBoost algorithm with early stopping rule.
pub mod booster;
/// A data loader with two independent caches. Alternatively, we use one
/// of the caches to feed data to the boosting algorithm, and the other
/// to load next sample set.
pub mod buffer_loader;

use commons::Model;
use commons::bins::Bins;
use commons::performance_monitor::PerformanceMonitor;
use config::Config;
use config::SampleMode;

use self::booster::Boosting;
use self::buffer_loader::BufferLoader;


pub fn start(config: &Config, sample_mode: &SampleMode, bins: &Vec<Bins>, init_tree: &Model) {
    debug!("Starting Scanner");
    let mut training_perf_mon = PerformanceMonitor::new();
    training_perf_mon.start();

    debug!("Starting the buffered loader.");
    let buffer_loader = BufferLoader::new(
        config.buffer_size,
        config.batch_size,
        sample_mode.clone(),
        config.sleep_duration,
        config.min_ess,
        config.exp_name.clone(),
    );
    debug!("Starting the booster.");
    let mut booster = Boosting::new(
        config.exp_name.clone(),
        init_tree.clone(),
        config.num_trees,
        config.num_splits,
        config.num_features,
        config.min_gamma,
        buffer_loader,
        bins.clone(),
        config.max_sample_size,
        config.default_gamma,
        config.save_process,
        config.save_interval,
    );
    booster.enable_network(config.local_name.clone(), config.port);
    booster.training(training_perf_mon.get_duration());
}