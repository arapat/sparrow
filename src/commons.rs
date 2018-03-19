extern crate time;

use self::time::PreciseTime;
use std::fmt;

use labeled_data::LabeledData;
use tree::Tree;

// TODO: use genetic types for reading data
type _TFeature = u8;
type _TLabel = u8;
pub type Example = LabeledData<_TFeature, _TLabel>;
pub type TLabel = _TLabel;
pub type Model = Vec<Tree>;
pub type ModelScore = (Model, f32);

pub type LossFunc = Fn(&Vec<(f32, f32)>) -> f32;

const DELTA: f32  = 0.0001;
const SHRINK: f32 = 0.8;
const THRESHOLD_FACTOR: f32 = 1.0;
const ALMOST_ZERO: f32 = 1e-8;


// Boosting related

#[inline]
pub fn get_weight(data: &Example, score: f32) -> f32 {
    // min(1.0, (-score * get_symmetric_label(data)).exp())
    (-score * get_symmetric_label(data)).exp()
}

pub fn get_weights(data: &Vec<Example>, scores: &[f32]) -> Vec<f32> {
    data.iter()
        .zip(scores.iter())
        .map(|(d, s)| get_weight(&d, *s))
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
pub fn max(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

#[inline]
pub fn min(a: f32, b: f32) -> f32 {
    if a > b {
        b
    } else {
        a
    }
}

#[inline]
pub fn is_zero(a: f32) -> bool {
    -ALMOST_ZERO < a && a < ALMOST_ZERO
}

#[inline]
pub fn is_positive(label: &f32) -> bool {
    is_zero(label - 1.0)
}


// Performance monitoring

pub struct PerformanceMonitor {
    start_time: PreciseTime,
    counter: usize
}

impl fmt::Debug for PerformanceMonitor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PerformanceMonitor")
    }
}

impl PerformanceMonitor {
    pub fn new() -> PerformanceMonitor {
        PerformanceMonitor {
            start_time: PreciseTime::now(),
            counter: 0
        }
    }

    pub fn start(&mut self) {
        self.start_time = PreciseTime::now();
        self.counter = 0;
    }

    pub fn update(&mut self, count: usize) {
        self.counter += count;
    }

    pub fn get_performance(&self) -> (usize, f32, f32) {
        let now = PreciseTime::now();
        let duration = 1e-3 * self.start_time.to(now).num_milliseconds() as f32;
        (self.counter, duration, (self.counter as f32) / duration)
    }

    pub fn get_duration(&self) -> f32 {
        let now = PreciseTime::now();
        1e-3 * self.start_time.to(now).num_milliseconds() as f32
    }
}
