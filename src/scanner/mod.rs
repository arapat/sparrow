/// The implementation of the AdaBoost algorithm with early stopping rule.
pub mod booster;
/// A data loader with two independent caches. Alternatively, we use one
/// of the caches to feed data to the boosting algorithm, and the other
/// to load next sample set.
pub mod buffer_loader; 