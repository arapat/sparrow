use commons::io::create_bufreader;

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

    /// minimum number of examples in a kd-tree node (grid)
    pub min_grid_size: usize,

    // Number of boosting iterations
    // pub num_iterations: usize,
    /// Number of decision trees (i.e. second-layer tree nodes)
    pub num_trees: usize,
    /// Maximum number of splits per tree
    pub num_splits: usize,

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


impl Config {
    pub fn new(config_filepath: &String) -> Config {
        serde_yaml::from_reader(create_bufreader(config_filepath)).unwrap()
    }
}


#[derive(Clone, Debug, PartialEq)]
pub enum SampleMode {
    // TODO: Support exchanging new sample directly to/from memory
    // MEMORY,
    LOCAL,
    S3,
}

impl SampleMode {
    pub fn new(sampling_mode: &String) -> SampleMode {
        match sampling_mode.to_lowercase().as_str() {
            "local"  => SampleMode::LOCAL,
            "s3"     => SampleMode::S3,
            _        => {
                error!("Unrecognized sampling mode. Use S3 by default.");
                SampleMode::S3
            }
        }
    }
}
