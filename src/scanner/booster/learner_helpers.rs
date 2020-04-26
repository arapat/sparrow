use rayon::prelude::*;

use Example;
use TFeature;
use commons::ExampleInSampleSet;
use commons::Model;
use commons::bins::Bins;
use scanner::buffer_loader::BufferLoader;

use commons::get_bound;

use super::learner::NUM_PREDS;
use super::learner::PREDS;
use super::learner::RuleStats;
use super::tree_node::TreeNode;


pub fn preprocess_data<'a>(
    data: &'a[ExampleInSampleSet], tree: &Model, expand_node: usize, rho_gamma: f32,
) -> Vec<(usize, f32, (&'a Example, RuleStats))> {
    data.par_iter().map(|(example, (weight, _, _, _))| {
        let labeled_weight = weight * (example.label as f32);
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
        (tree.get_leaf_index_prediction(expand_node, example), *weight, (example, vals))
    }).collect()
}

// if `total_weight` put into account those examples that a node abstained, the comparison
// is then among all 'specialists'.
pub fn find_tree_node<'a>(
    data: &'a Vec<(f32, (&Example, RuleStats))>, feature_index: usize,
    rho_gamma: f32, count: usize, total_weight: f32, total_weight_sq: f32, expand_node: usize,
    bin: &'a Bins, weak_rules_score: &'a mut Vec<[f32; 2]>, sum_c_squared: &'a mut Vec<[f32; 2]>,
    debug_info: (((&'a mut Vec<f32>, &'a mut Vec<f32>), &'a mut Vec<f32>), &'a mut Vec<f32>),
) -> Option<TreeNode> {
    let (((num_positive, num_negative), weight_positive), weight_negative) = debug_info;

    // <Split, NodeId, RuleId, stats, LeftOrRight>
    // the last element of is for the examples that are larger than all split values
    let mut bin_accum_vals: Vec<RuleStats> =
        vec![[[(0.0, 0.0); 2]; NUM_PREDS]; bin.len() + 1];
    // Counts the total weights and the counts for both positive and negative examples
    let mut counts: [usize; 2] = [0, 0];
    let mut weights: [f32; 2]  = [0.0, 0.0];
    data.iter()
        .for_each(|(w, (example, vals))| {
            let flip_index = example.feature[feature_index] as usize;
            let t = &mut bin_accum_vals[flip_index];
            for j in 0..NUM_PREDS {
                for k in 0..t[j].len() {
                    t[j][k].0 += vals[j][k].0;
                    t[j][k].1 += vals[j][k].1;
                }
            }
            if example.label > 0 {
                counts[0]  += 1;
                weights[0] += w;
            } else {
                counts[1]  += 1;
                weights[1] += w;
            }
        });

    let mut accum_left  = [[0.0; 2]; NUM_PREDS];
    let mut accum_right = [[0.0; 2]; NUM_PREDS];
    // Accumulate sum of the stats of all examples that go to the right child
    for j in 0..bin.len() { // Split value
        for pred_idx in 0..NUM_PREDS { // Types of rule
            for it in 0..accum_right[pred_idx].len() {
                accum_right[pred_idx][it] +=
                    bin_accum_vals[j][pred_idx][it].1;
            }
        }
    }
    // Now update each splitting values of the bin
    let mut valid_weak_rule = None;
    (0..bin.len()).for_each(|j| {
        for pred_idx in 0..NUM_PREDS { // Types of rule
            // Move examples from the right to the left child
            for it in 0..accum_left[pred_idx].len() {
                accum_left[pred_idx][it]  +=
                    bin_accum_vals[j][pred_idx][it].0;
                accum_right[pred_idx][it] -=
                    bin_accum_vals[j][pred_idx][it].1;
            }
            let accum: Vec<f32> = accum_left[pred_idx].iter()
                                                        .zip(accum_right[pred_idx].iter())
                                                        .map(|(a, b)| *a + *b)
                                                        .collect();
            {
                let rule_idx = pred_idx;
                let weak_rules_score =
                    &mut weak_rules_score[j][rule_idx];
                let sum_c_squared    = &mut sum_c_squared[j][rule_idx];
                let num_positive = &mut num_positive[j];
                let num_negative = &mut num_negative[j];
                let weight_positive = &mut weight_positive[j];
                let weight_negative = &mut weight_negative[j];

                *weak_rules_score   += accum[0];
                *sum_c_squared      += accum[1];
                *num_positive       += counts[0] as f32;
                *num_negative       += counts[1] as f32;
                *weight_positive    += weights[0];
                *weight_negative    += weights[1];
                // Check stopping rule
                let sum_c = *weak_rules_score - 2.0 * rho_gamma * total_weight;
                let sum_c_squared = *sum_c_squared +
                    4.0 * rho_gamma * rho_gamma * total_weight_sq;
                let bound = get_bound(sum_c, sum_c_squared);
                if sum_c > bound {
                    let base_pred = 0.5 * (
                        (0.5 + rho_gamma) / (0.5 - rho_gamma)
                    ).ln();
                    let real_pred =
                        (base_pred * PREDS[pred_idx].0, base_pred * PREDS[pred_idx].1);
                    valid_weak_rule = Some(
                        TreeNode {
                            prt_index:      expand_node,
                            feature:        feature_index,
                            threshold:      j as TFeature,
                            predict:        real_pred,

                            gamma:          rho_gamma,
                            raw_martingale: *weak_rules_score,
                            sum_c:          sum_c,
                            sum_c_squared:  sum_c_squared,
                            bound:          bound,
                            num_scanned:    count,

                            positive:        *num_positive as usize,
                            negative:        *num_negative as usize,
                            positive_weight: *weight_positive,
                            negative_weight: *weight_negative,

                            fallback:       false,
                        }
                    );
                }
            }
        }
    });
    valid_weak_rule
}


pub fn gen_tree_node(
    expand_node_index: usize, feature_index: usize, bin_index: usize, rule_index: usize, ratio: f32,
) -> TreeNode {
    let rho_gamma = ratio / 2.0;
    let base_pred = 0.5 * (
        (0.5 + rho_gamma) / (0.5 - rho_gamma)
    ).ln();
    let real_pred =
        (base_pred * PREDS[rule_index].0, base_pred * PREDS[rule_index].1);
    TreeNode {
        prt_index:      expand_node_index,
        feature:        feature_index,
        threshold:      bin_index as TFeature,
        predict:        real_pred,
        gamma:          rho_gamma,

        fallback:        true,

        // other attributes are for debugging purpose only
        raw_martingale: 0.0,
        sum_c:          0.0,
        sum_c_squared:  0.0,
        bound:          0.0,
        num_scanned:    0,

        positive:        0,
        negative:        0,
        positive_weight: 0.0,
        negative_weight: 0.0,
    }
}


pub fn get_base_node(max_sample_size: usize, data_loader: &mut BufferLoader) -> (f32, f32, f32) {
    let mut sample_size = max_sample_size;
    let mut n_pos = 0;
    let mut n_neg = 0;
    while sample_size > 0 {
        let (data, _) = data_loader.get_next_batch(true);
        let (num_pos, num_neg) =
            data.par_iter().fold(
                || (0, 0),
                |(num_pos, num_neg), (example, _)| {
                    if example.label > 0 {
                        (num_pos + 1, num_neg)
                    } else {
                        (num_pos, num_neg + 1)
                    }
                }
            ).reduce(|| (0, 0), |(a1, a2), (b1, b2)| (a1 + b1, a2 + b2));
        n_pos += num_pos;
        n_neg += num_neg;
        sample_size -= data.len();
    }

    let gamma = (0.5 - n_pos as f32 / (n_pos + n_neg) as f32).abs();
    let prediction = 0.5 * (n_pos as f32 / n_neg as f32).ln();
    info!("root-tree-info, {}, {}, {}, {}", 1, max_sample_size, gamma, gamma * gamma);
    (gamma, prediction, gamma)
}
