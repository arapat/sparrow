use rayon::prelude::*;

use Example;
use commons::ExampleInSampleSet;
use commons::Model;

use super::learner::NUM_PREDS;
use super::learner::PREDS;
use super::learner::RuleStats;

pub fn preprocess_data<'a>(
    data: &'a[ExampleInSampleSet], tree: &Model, expand_node: usize, rho_gamma: f32,
) -> Vec<(f32, (&'a Example, RuleStats))> {
    data.par_iter().map(|(example, (weight, _, _, _))| {
        let labeled_weight = weight * (example.label as f32);
        // let null_weight = 2.0 * rho_gamma * weight;
        let null_weight = 2.0 * rho_gamma * weight;
        let mut vals: RuleStats = [[(0.0, 0.0); 2]; NUM_PREDS];
        PREDS.iter().enumerate().for_each(|(i, pred)| {
            let abs_val = (pred.0 * labeled_weight, pred.1 * labeled_weight);
            let ci      = (abs_val.0 - null_weight, abs_val.1 - null_weight);
            vals[i][0]  = abs_val;
            vals[i][1]  = (
                ci.0 * ci.0 - null_weight * null_weight,
                ci.1 * ci.1 - null_weight * null_weight
            );
        });
        (tree.is_visited(example, expand_node), *weight, (example, vals))
    }).filter(|(contains, _, _)| *contains)
        .map(|(_, w, b)| (w, b))
        .collect()
}
