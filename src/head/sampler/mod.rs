/// A stratified storage structor that organize examples on disk according to their weights.
pub mod stratified_storage;


use std::sync::Arc;
use std::sync::RwLock;
use std::sync::mpsc::Sender;

use commons::channel::Receiver;
use commons::Model;
use commons::bins::Bins;
use commons::INIT_MODEL_PREFIX;
use config::Config;
use config::SampleMode;

use self::stratified_storage::StratifiedStorage;
use commons::packet::TaskPacket;


pub fn start_sampler_async(
    config: &Config,
    sample_mode: &SampleMode,
    bins: &Vec<Bins>,
    init_tree: &Model,
    next_model_recv: Receiver<(Model, String)>,
    packet_sender: Sender<(Option<String>, TaskPacket)>,
) -> Arc<RwLock<bool>> {
    debug!("Starting Sampler");
    let sampler_state = Arc::new(RwLock::new(true));
    debug!("Starting the stratified structure.");
    let init_model_name = INIT_MODEL_PREFIX.to_string();
    // start the sampling process in the stratified storage
    let stratified_structure = StratifiedStorage::new(
        init_tree.clone(),
        init_model_name.clone(),
        config.num_examples,
        config.buffer_size,
        config.num_features,
        config.positive.clone(),
        config.num_examples_per_block,
        config.disk_buffer_filename.as_ref(),
        sample_mode.clone(),
        config.num_assigners,
        config.num_samplers,
        next_model_recv,
        config.channel_size,
        sampler_state.clone(),
        config.debug_mode,
        config.resume_training,
        config.exp_name.clone(),
        packet_sender,
    );

    debug!("Initializing the stratified structure.");
    stratified_structure.init_stratified_from_file(
        config.training_filename.clone(),
        config.num_examples,
        config.batch_size,
        config.num_features,
        bins.clone(),
        init_tree.clone(),
    );

    sampler_state
}