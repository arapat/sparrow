pub mod bins;
pub mod channel;
pub mod io;
pub mod model;
// The class of the training examples.
pub mod labeled_data;
pub mod packet;
pub mod performance_monitor;
// Functions to read/write samples, models, and worker assignments
pub mod persistent_io;
// The class of the weak learner, namely a decision stump.
pub mod tree;
// helper function for testing
#[cfg(test)] pub mod test_helper;

use rayon::prelude::*;
use std::f32::INFINITY;

use super::Example;

// current score and size, base model and size
pub type ExampleInSampleSet = (Example, (f32, f32, usize, usize));  // weight, score, new_ver, base_ver
pub type ExampleWithScore = (Example, (f32, usize));

const DELTA: f32  = 0.000001;
const SHRINK: f32 = 1.0;
const THRESHOLD_FACTOR: f32 = 1.0;
const ALMOST_ZERO: f32 = 1e-8;

// Boosting related

#[inline]
pub fn get_weight(data: &Example, score: f32) -> f32 {
    // min(1.0, (-score * data.label).exp())
    (-score * (data.label as f32)).exp()
}

#[allow(dead_code)]
#[inline]
pub fn get_weights(data: &Vec<Example>, scores: &[f32]) -> Vec<f32> {
    data.par_iter()
        .zip(scores.par_iter())
        .map(|(d, s)| get_weight(&d, *s))
        .collect()
}

#[inline]
pub fn get_bound(sum_c: f32, sum_c_squared: f32) -> f32 {
    let threshold: f32 = THRESHOLD_FACTOR * 173.0 * (4.0 / DELTA).ln();
    if sum_c_squared >= threshold {
        let log_log_term = 3.0 * sum_c_squared / 2.0 / sum_c.abs();
        let log_log = {
            if log_log_term > 2.7183 {
                log_log_term.ln().ln()
            } else {
                0.0
            }
        };
        SHRINK * (
            3.0 * sum_c_squared * (2.0 * log_log + (2.0 / DELTA).ln())
        ).sqrt()
    } else {
        INFINITY
    }
}

/// Set initial weights to the samples
#[inline]
pub fn set_init_weight(examples: Vec<ExampleWithScore>) -> Vec<ExampleInSampleSet> {
    examples.into_iter()
            .map(|t| {
                let (example, (_score, model_size)) = t;
                // sampling weights are ignored
                let w = get_weight(&example, 0.0);
                (example.clone(), (w, 0.0, model_size, model_size))
            }).collect()
}

// Computational functions

#[inline]
pub fn is_zero(a: f32) -> bool {
    get_sign(a as f64) == 0
}

#[inline]
pub fn get_sign(a: f64) -> i8 {
    if a < -ALMOST_ZERO as f64 {
        -1
    } else if a > ALMOST_ZERO as f64 {
        1
    } else {
        0
    }
}
