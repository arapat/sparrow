use rayon::prelude::*;
use time::PreciseTime;

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
const SHRINK: f32 = 1.0;
const THRESHOLD_FACTOR: f32 = 6.0;
const ALMOST_ZERO: f32 = 1e-8;


// Boosting related

#[inline]
pub fn get_weight(data: &Example, score: f32) -> f32 {
    min(1.0, (-score * get_symmetric_label(data)).exp())
    // (-score * get_symmetric_label(data)).exp()
}

pub fn get_weights(data: &Vec<Example>, scores: &[f32]) -> Vec<f32> {
    data.par_iter()
        .zip(scores.par_iter())
        .map(|(d, s)| get_weight(&d, *s))
        .collect()
}

#[inline]
pub fn get_bound(sum_c: &f32, sum_c_squared: &f32) -> Option<f32> {
    // loglogv will be np.nan if conditons are not satisfied
    let threshold: f32 = THRESHOLD_FACTOR * 173.0 * (4.0 / DELTA).ln();
    if *sum_c_squared >= threshold {
        let log_log_term = 5.0 * sum_c_squared / 2.0 / sum_c.abs();
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
    -ALMOST_ZERO < a && a < ALMOST_ZERO
}

#[inline]
pub fn is_positive(label: &f32) -> bool {
    is_zero(label - 1.0)
}


// Performance monitoring
#[derive(PartialEq)]
enum PerformanceMonitorStatus {
    PAUSE,
    RUNNING
}

pub struct PerformanceMonitor {
    start_time: PreciseTime,
    last_check: PreciseTime,
    status: PerformanceMonitorStatus,
    duration: i64,
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
            last_check: PreciseTime::now(),
            status: PerformanceMonitorStatus::PAUSE,
            duration: 0,
            counter: 0,
        }
    }

    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.duration = 0;
        self.counter = 0;
        self.status = PerformanceMonitorStatus::PAUSE;
        self.last_check = PreciseTime::now();
    }

    pub fn start(&mut self) {
        self.resume();
    }

    pub fn resume(&mut self) {
        assert!(self.status == PerformanceMonitorStatus::PAUSE);
        self.start_time = PreciseTime::now();
        self.status = PerformanceMonitorStatus::RUNNING;
    }

    pub fn update(&mut self, count: usize) {
        assert!(self.status == PerformanceMonitorStatus::RUNNING);
        self.counter += count;
    }

    pub fn pause(&mut self) {
        assert!(self.status == PerformanceMonitorStatus::RUNNING);
        self.duration += self.start_time.to(PreciseTime::now()).num_microseconds().unwrap();
        self.status = PerformanceMonitorStatus::PAUSE;
    }

    pub fn get_performance(&mut self) -> (i64, usize, f32, f32) {
        let since_last_check = self.last_check.to(PreciseTime::now()).num_seconds();
        let duration = self.get_duration();
        (since_last_check, self.counter, duration, (self.counter as f32) / duration)
    }

    pub fn reset_last_check(&mut self) {
        self.last_check = PreciseTime::now();
    }

    pub fn get_duration(&self) -> f32 {
        let microseconds = self.duration +
            if self.status == PerformanceMonitorStatus::RUNNING {
                self.start_time.to(PreciseTime::now()).num_microseconds().unwrap()
            } else {
                0
            };
        1e-6 * microseconds as f32
    }
}
