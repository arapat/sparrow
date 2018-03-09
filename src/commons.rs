use labeled_data::LabeledData;

// TODO: use genetic types for reading data
type _TFeature = f32;
type _TLabel = f32;
pub type Example = LabeledData<_TFeature, _TLabel>;


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
pub fn max(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}
