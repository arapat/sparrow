pub mod performance_monitor;
pub mod io;

use rayon::prelude::*;

use labeled_data::LabeledData;
use tree::Tree;

// TODO: use genetic types for reading data
pub type TFeature = u8;
pub type TLabel = u8;
pub type Example = LabeledData<TFeature, TLabel>;
pub type ExampleInSampleSet = (Example, (f32, usize), (f32, usize));
pub type ExampleWithScore = (Example, (f32, usize));
pub type Model = Vec<Tree>;
pub type ModelScore = (Model, f32);

pub type LossFunc = Fn(&Vec<(f32, f32)>) -> f32;

const DELTA: f32  = 0.0001;
const SHRINK: f32 = 1.0;
const THRESHOLD_FACTOR: f32 = 6.0;
const ALMOST_ZERO: f32 = 1e-8;


// Boosting related

#[inline]
pub fn get_weight(data: &Example, score: f32) -> f32 {
    // min(1.0, (-score * get_symmetric_label(data)).exp())
    (-score * get_symmetric_label(data)).exp()
}

pub fn get_weights(data: &Vec<Example>, scores: &[f32]) -> Vec<f32> {
    data.par_iter()
        .zip(scores.par_iter())
        .map(|(d, s)| get_weight(&d, *s))
        .collect()
}

pub fn get_absolute_weights(data: &[ExampleInSampleSet]) -> Vec<f32> {
    data.par_iter()
        .map(|(d, _, (s, _))| get_weight(&d, *s))
        .collect()
}

pub fn get_relative_weights(data: &[ExampleInSampleSet]) -> Vec<f32> {
    data.par_iter()
        .map(|(d, (s1, _), (s2, _))| get_weight(&d, s2 - s1))
        .collect()
}

#[inline]
pub fn get_bound(sum_c: &f32, sum_c_squared: &f32) -> Option<f32> {
    // loglogv will be np.nan if conditons are not satisfied
    let threshold: f32 = THRESHOLD_FACTOR * 173.0 * (4.0 / DELTA).ln();
    if *sum_c_squared >= threshold {
        let log_log_term = 3.0 * sum_c_squared / 2.0 / sum_c.abs();
        if log_log_term > 1.0 {
            let log_log = log_log_term.ln().ln();
            let sqrt_term = 3.0 * sum_c_squared * (2.0 * log_log + (2.0 / DELTA).ln());
            if sqrt_term >= 0.0 {
                return Some(SHRINK * sqrt_term.sqrt());
            }
        }
    }
    None
}

#[inline]
pub fn get_symmetric_label(data: &Example) -> f32 {
    let label = *data.get_label() as f32;
    if is_positive(&label) {
        1.0
    } else {
        -1.0
    }
}


// Computational functions

#[inline]
pub fn max<T>(a: T, b: T) -> T where T: PartialOrd {
    if a > b {
        a
    } else {
        b
    }
}

#[inline]
pub fn min<T>(a: T, b: T) -> T where T: PartialOrd {
    if a > b {
        b
    } else {
        a
    }
}

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

#[inline]
pub fn is_positive(label: &f32) -> bool {
    is_zero(label - 1.0)
}