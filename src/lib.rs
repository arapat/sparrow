/*!
rustboost is an implementation of TMSN for boosting.
*/
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate log;
#[macro_use] extern crate chan;
extern crate bincode;
extern crate bufstream;
extern crate ordered_float;
extern crate rand;
extern crate rayon;
extern crate serde_json;
extern crate threadpool;
extern crate time;

/// The values of percentiles on each feature dimension.
pub mod bins;
/// High level implementation of the AdaBoost algorithm with early stopping rule.
pub mod boosting;
/// Common functions and classes.
pub mod commons;
/// Configurations that can be stored in a separate JSON file.
pub mod config;
/// A data loader with two independent caches. Alternatively, we use one
/// of the caches to feed data to the boosting algorithm, and the other
/// to load next sample set.
pub mod buffer_loader; 
/// The class of the training examples.
pub mod labeled_data;
/// The class that evaluates the early stopping rule on the candidate weak rules.
pub mod learner;
/// Manage network I/O
pub mod network;
/// Sampling data from the stratified storage.
pub mod sampler;
/// A stratified storage structor that organize examples on disk according to their weights.
pub mod stratified_storage;
/// The class of the weak learner, namely a decision stump.
pub mod tree;
// mod validator;