# Sparrow

Sparrow is an impementation of boosting that is optimized for training on very large datasets, as well as training in limited memory settings. The opimizations involve two technologies: early stopping and selective sampling. Please read [our paper](https://arxiv.org/abs/1901.09047) for more details.

## Prerequisites

1. Install Rust following the instruction on [the Rust offical webiste](https://www.rust-lang.org/tools/install).
2. Compile Sparrow: `cargo build --release`

The Sparrow binary file would be generated at `target/release/sparrow`.

## Usage

Sparrow is written as a Rust library. It also supports running as a binary. The sparrow binary reads configuration from a specified configuration file. Many examples of the configuration files can be found in the [examples/](examples) directory.

To run the Sparrow binary, please provide the path to the configuration file.

For training,
```bash
./sparrow train <path to the config file>
```

For testing (or prediction),
```bash
./sparrow test <path to the config file>
```


### Fields in the Config File

<dl>

<dt>training_filename:</dt>
<dd>File path to the training data</dd>

<dt>num_examples:</dt>
<dd>Number of training examples</dd>

<dt>num_features:</dt>
<dd>Number of features</dd>

<dt>range:</dt>
<dd>Range of the features for creating weak rules</dd>

<dt>positive:</dt>
<dd>Label for positive examples</dd>

<dt>testing_filename:</dt>
<dd>File path to the testing data</dd>

<dt>num_testing_examples:</dt>
<dd>Number of testing examples</dd>

<dt>max_sample_size:</dt>
<dd>Number of examples to scan for generating heuristic used in Sparrow</dd>

<dt>max_bin_size:</dt>
<dd>Maximum number of bins for discretizing continous feature values</dd>

<dt>min_gamma:</dt>
<dd>Minimum value of the \gamma of the generated tree nodes</dd>

<dt>default_gamma:</dt>
<dd>Default maximum value of the \gamma for generating tree nodes</dd>

<dt>max_trials_before_shrink:</dt>
<dd>Maximum number of examples to scan before shrinking the value of \gamma</dd>

<dt>min_ess:</dt>
<dd>Minimum effective sample size for triggering resample</dd>

<dt>num_iterations:</dt>
<dd>Number of boosting iterations</dd>

<dt>max_leaves:</dt>
<dd>Maximum number of tree leaves in each boosted tree</dd>

<dt>channel_size:</dt>
<dd>Maximum number of elements in the channel connecting scanner and sampler</dd>

<dt>buffer_size:</dt>
<dd>Number of examples in the sample set that needs to be loaded into memory</dd>

<dt>batch_size:</dt>
<dd>Number of examples to process in each weak rule updates</dd>

<dt>serial_sampling:</dt>
<dd>Set to true to stop running sampler in the background of the scanner</dd>

<dt>num_examples_per_block:</dt>
<dd>Number of examples in a block on the stratified binary file</dd>

<dt>disk_buffer_filename:</dt>
<dd>File name for the stratified binary file</dd>

<dt>num_assigners:</dt>
<dd>Number of threads for putting examples back to correct strata</dd>

<dt>num_samplers:</dt>
<dd>Number of threads for sampling examples from strata</dd>

<dt>network:</dt>
<dd>IP addresses of other machines in the network</dd>

<dt>port:</dt>
<dd>The network port used for parallel training</dd>

<dt>local_name:</dt>
<dd>Identifier for the local machine</dd>

<dt>save_process:</dt>
<dd>Flag for keeping all intermediate models during training (for debugging purpose)</dd>

<dt>save_interval:</dt>
<dd>Number of iterations between persisting models on disk</dd>

<dt>debug_mode:</dt>
<dd>Flag for activating debug mode</dd>

<dt>models_table_filename:</dt>
<dd>(for validation only) the file names of the models to run the validation</dd>

<dt>incremental_testing:</dt>
<dd>Flag indicating if models are trained incrementally</dd>

<dt>testing_scores_only:</dt>
<dd>Flag for validation mode, set to true to output raw scores of testing examples, and set to false for printing the validation scores but not raw scores</dd>

</dl>
