use rand;
use rand::Rng;

use TFeature;
use commons::ExampleWithScore;
use commons::labeled_data::LabeledData;


#[allow(dead_code)]
pub fn get_synthetic_example(features: Vec<TFeature>, label: i8, score: f32) -> ExampleWithScore {
    let example = LabeledData::new(features, label);
    (example, (score, 0))
}


#[allow(dead_code)]
pub fn get_n_random_examples(n: usize, num_features: usize) -> Vec<ExampleWithScore> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| {
        let features: Vec<TFeature> = (0..num_features).map(|_| { rng.gen::<TFeature>() })
                                                       .collect();
        let label: i8 = rng.gen_range(0, 1);
        get_synthetic_example(features, label, 0.0)
    }).collect()
}
