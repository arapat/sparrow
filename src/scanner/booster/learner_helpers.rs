use rayon::prelude::*;

use Example;
use TFeature;
use commons::ExampleInSampleSet;
use commons::Model;
use commons::bins::Bins;
use scanner::buffer_loader::BufferLoader;

use commons::get_bound;

use super::learner::NUM_RULES;
use super::learner::PREDS;
use super::learner::RuleStats;
use super::learner::EarlyStoppingIntermediate;
use super::learner::EarlyStoppingStatsAtThreshold;
use super::tree_node::TreeNode;


pub fn preprocess_data<'a>(
    data: &'a[ExampleInSampleSet], tree: &Model, expand_node: usize, rho_gamma: f32,
) -> Vec<(usize, &'a Example, f32, RuleStats)> {
    data.par_iter().map(|(example, (weight, _, _, _))| {
        let leaf_index = tree.get_leaf_index_prediction(expand_node, example);
        let rule_stats: RuleStats = PREDS.iter().map(|pred| {
            EarlyStoppingIntermediate::new(pred, example, weight, rho_gamma)
        }).collect();
        (leaf_index, example, *weight, rule_stats)
    }).collect()
}

// if `total_weight` put into account those examples that a node abstained, the comparison
// is then among all 'specialists'.
pub fn find_tree_node<'a>(
    data: &'a Vec<(&Example, f32, RuleStats)>, feature_index: usize,
    rho_gamma: f32, count: usize, total_weight: f32, total_weight_sq: f32, expand_node: usize,
    feature_bin: &'a Bins, stats_feature: &'a mut Vec<EarlyStoppingStatsAtThreshold>,
) -> Option<TreeNode> {
    // <Split, NodeId, RuleId, stats, LeftOrRight>
    // the last element of is for the examples that are larger than all split values
    let mut intermediate_stats: Vec<RuleStats> =
        (0..(feature_bin.len() + 1)).map(|_| {
            (0..NUM_RULES).map(|_| EarlyStoppingIntermediate::zero()).collect()
        }).collect();
    let mut counts: [usize; 2] = [0, 0];  // positive and negative
    let mut weights: [f32; 2]  = [0.0, 0.0];  // ditto
    let mut weight_sq: f32     = 0.0;
    data.iter()
        .for_each(|(example, w, example_intermediate_stats)| {
            let flip_index = example.feature[feature_index] as usize;
            intermediate_stats[flip_index].iter_mut().zip(
                example_intermediate_stats
            ).for_each(|(stats, elem)| {
                stats.add(elem);
            });
            weight_sq += w * w;
            if example.label > 0 {
                counts[0]  += 1;
                weights[0] += w;
            } else {
                counts[1]  += 1;
                weights[1] += w;
            }
        });

    let mut accum_stats: RuleStats = (0..NUM_RULES).map(|_| {
        EarlyStoppingIntermediate::zero()
    }).collect();
    // Accumulate sum of the stats of all examples that go to the right child
    intermediate_stats.iter().for_each(|elems| {
        accum_stats.iter_mut()
                   .zip(elems.iter())
                   .for_each(|(accum, elem)| {
                       accum.add_right(elem);
                    });
    });

    // Now update each splitting threshold of the feature
    let mut valid_weak_rule = None;
    stats_feature.iter_mut().enumerate()
                 .zip(intermediate_stats.iter())
                 .for_each(|((thr_index, sum_martingale), thr_stats)| {
                     // node-level stats, it is redundant but avoid another aggregator
                     sum_martingale.num_positive                += counts[0];
                     sum_martingale.num_negative                += counts[1];
                     sum_martingale.weight_positive             += weights[0];
                     sum_martingale.weight_negative             += weights[1];
                     sum_martingale.weight_squared              += weight_sq;
                     let node_weight_sum =
                         sum_martingale.weight_positive + sum_martingale.weight_negative;
                     // iterate over each threshold, now process each rule
                     accum_stats.iter_mut().zip(thr_stats.iter()).zip(PREDS.iter())
                         .for_each(|((batch_stats, stats_at_thr), preds)| {
                             // iterate over each prediction rule

                             // move examples from the right to the left child
                             batch_stats.move_to_left(stats_at_thr);
                             let batch_score = batch_stats.left_score + batch_stats.right_score;
                             let batch_edge = batch_stats.left_c + batch_stats.right_c;
                             let batch_edge_squared =
                                 batch_stats.left_c_squared + batch_stats.right_c_squared;
                             {
                                 sum_martingale.weak_rules_score[thr_index] += batch_score;
                                 sum_martingale.sum_c[thr_index]            += batch_edge;
                                 sum_martingale.sum_c_squared[thr_index]    += batch_edge_squared;

                                 // Check stopping rule
                                 // let sum_c = *weak_rules_score - 2.0 * rho_gamma * total_weight;
                                 let abstained_weight = total_weight - node_weight_sum;
                                 let sum_c = sum_martingale.sum_c[thr_index] -
                                    2.0 * rho_gamma * abstained_weight;
                                 let abstained_weight_squared =
                                    total_weight_sq - sum_martingale.weight_squared;
                                 let sum_c_squared = sum_martingale.sum_c_squared[thr_index]
                                    + 4.0 * rho_gamma * rho_gamma * abstained_weight_squared;
                                 let bound = get_bound(sum_c, sum_c_squared);
                                 if sum_c > bound {
                                     let base_pred = 0.5 *
                                        ((0.5 + rho_gamma) / (0.5 - rho_gamma)).ln();
                                     let real_pred = (base_pred * preds.0, base_pred * preds.1);
                                     valid_weak_rule = Some(
                                         TreeNode {
                                             prt_index:      expand_node,
                                             feature:        feature_index,
                                             threshold:      thr_index as TFeature,
                                             predict:        real_pred,

                                             gamma:          rho_gamma,
                                             raw_martingale: sum_martingale.weak_rules_score[thr_index],
                                             sum_c:          sum_martingale.sum_c[thr_index],
                                             sum_c_squared:  sum_martingale.sum_c_squared[thr_index],
                                             bound:          bound,
                                             num_scanned:    count,

                                             positive:        sum_martingale.num_positive,
                                             negative:        sum_martingale.num_negative,
                                             positive_weight: sum_martingale.weight_positive,
                                             negative_weight: sum_martingale.weight_negative,

                                             fallback:       false,
                                         }
                                     );
                                 }
                             }
                         });
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
