use labeled_data::LabeledData;

// TODO: use genetic types for reading data
type _TFeature = f32;
type _TLabel = f32;
pub type Example = LabeledData<_TFeature, _TLabel>;

const DELTA: f32  = 0.0001;
const SHRINK: f32 = 0.8;


// Boosting related

#[inline]
pub fn get_weight(data: &Example, score: f32) -> f32 {
    max(1.0, (-score * data.get_label()).exp())
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
    let threshold: f32 = 173.0 * (4.0 / DELTA).ln();
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


// Computational functions

#[inline]
pub fn max(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}
