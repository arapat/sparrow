use rayon::prelude::*;

use commons::ExampleInSampleSet;
use super::super::Example;
use super::super::TFeature;

use buffer_loader::BufferLoader;
use commons::bins::Bins;
use commons::get_bound;
use commons::is_zero;
use commons::Model;

// TODO: The tree generation and score updates are for AdaBoost only,
// extend it to other potential functions

/*
TODO: extend support to regression tasks
TODO: re-use ScoreBoard space of the generated rules to reduce the memory footprint by half
      (just adding a mapping from the index here to the index in the tree should do the job)

ScoreBoard structure:
Feature index -> split value index -> candidate splitting node index (removed) -> prediction types

Each split corresponds to 2 types of predictions,
    1. Left +1, Right -1;
    2. Left -1, Right +1;
*/
const NUM_PREDS: usize = 2;
const NUM_RULES: usize = NUM_PREDS;
const PREDS: [(f32, f32); NUM_PREDS] = [(-1.0, 1.0), (1.0, -1.0)];
// (pred1, false), (pred1, true), (pred2, false), (pred2, true)
type ScoreBoard = Vec<Vec<[f32; NUM_RULES]>>;
type ScoreBoard1 = Vec<Vec<f32>>;
// (f32, f32) -> Stats if falls under left, and if falls under right
type RuleStats = [[(f32, f32); 2]; NUM_PREDS];


/// A weak rule with an edge larger or equal to the targetting value of `gamma`
pub struct TreeNode {
    pub prt_index: usize,
    pub feature: usize,
    pub threshold: TFeature,
    pub predict: (f32, f32),

    pub gamma: f32,
    raw_martingale: f32,
    sum_c: f32,
    sum_c_squared: f32,
    bound: f32,
    pub num_scanned: usize,
    pub fallback: bool,

    pub positive: usize,
    pub negative: usize,
    pub positive_weight: f32,
    pub negative_weight: f32,
}

impl TreeNode {
    pub fn write_log(&self) {
        info!(
            "tree-node-info, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
            self.prt_index,
            self.feature,
            self.threshold,
            self.predict.0,
            self.predict.1,

            self.num_scanned,
            self.gamma,
            self.raw_martingale,
            self.sum_c,
            self.sum_c_squared,
            self.bound,
            self.sum_c_squared / self.num_scanned as f32,

            self.positive,
            self.negative,
            self.positive_weight,
            self.negative_weight,

            self.fallback,
        );
    }
}


/// Statisitics of all weak rules that are being evaluated.
/// The objective of `Learner` is to find a weak rule that satisfies the condition of
/// the stopping rule.
// [i][k][j][rule_id] => bin i slot j candidate-node k
#[derive(Serialize, Deserialize)]
pub struct Learner {
    bins: Vec<Bins>,
    num_features:   usize,
    _default_gamma: f32,
    min_gamma: f32,

    pub rho_gamma:        f32,
    pub root_gamma:       f32,
    pub expand_node:      usize,
    // global trackers
    pub total_count:  usize,
    total_weight:     f32,
    total_weight_sq:  f32,
    // trackers for each candidate
    weak_rules_score: ScoreBoard,
    sum_c_squared:    ScoreBoard,
    // trackers for debugging
    num_positive:     ScoreBoard1,
    num_negative:     ScoreBoard1,
    weight_positive:  ScoreBoard1,
    weight_negative:  ScoreBoard1,
    // track the number of examples exposed to each splitting candidates
    count:            usize,
    sum_weight:       f32,
}

impl Learner {
    /// Create a `Learner` that search for valid weak rules.
    /// `default_gamma` is the initial value of the edge `gamma`.
    /// `bins` is vectors of the all thresholds on all candidate features for generating weak rules.
    pub fn new(
        min_gamma: f32,
        default_gamma: f32,
        num_features: usize,
        bins: Vec<Bins>,
    ) -> Learner {
        let mut learner = Learner {
            bins: bins,
            num_features: num_features.clone(),
            _default_gamma: default_gamma.clone(),
            min_gamma: min_gamma,

            rho_gamma:        default_gamma.clone(),
            root_gamma:       default_gamma.clone(),
            expand_node:      0,
            total_count:      0,
            total_weight:     0.0,
            total_weight_sq:  0.0,
            weak_rules_score: vec![],
            sum_c_squared:    vec![],

            num_positive:     vec![],
            num_negative:     vec![],
            weight_positive:  vec![],
            weight_negative:  vec![],

            count:            0,
            sum_weight:      0.0,
        };
        for i in 0..num_features {
            let size = learner.bins[i].len();
            learner.weak_rules_score.push(vec![[0.0; NUM_RULES]; size]);
            learner.sum_c_squared.push(vec![[0.0; NUM_RULES]; size]);
            learner.num_positive.push(vec![0.0; size]);
            learner.num_negative.push(vec![0.0; size]);
            learner.weight_positive.push(vec![0.0; size]);
            learner.weight_negative.push(vec![0.0; size]);
        }
        learner
    }

    /// Reset the statistics of all candidate weak rules
    /// (except gamma, because the advantage of the root node is not likely to increase)
    /// Trigger when the model or the gamma is changed
    pub fn reset(&mut self) {
        for i in 0..self.num_features {
            for j in 0..self.bins[i].len() {
                for k in 0..NUM_RULES {
                    self.weak_rules_score[i][j][k] = 0.0;
                    self.sum_c_squared[i][j][k]    = 0.0;
                }
                self.num_positive[i][j] = 0.0;
                self.num_negative[i][j] = 0.0;
                self.weight_positive[i][j] = 0.0;
                self.weight_negative[i][j] = 0.0;
            }
        }
        debug!("learner, learner is reset, {}, {}, {}, {}, {}",
               self.expand_node, self.total_count, self.total_weight, self.count, self.sum_weight);
        self.sum_weight = 0.0;
        self.count = 0;
        self.total_count = 0;
        self.total_weight = 0.0;
        self.total_weight_sq = 0.0;
    }

    /*
    fn get_max_empirical_ratio(&self) -> (f32, (usize, usize, usize)) {
        let mut max_ratio = 0.0;
        let mut actual_ratio = 0.0;
        let mut rule_id = (0, 0, 0, 0, 0);
        for i in 0..self.num_features {
            for j in 0..self.bins[i].len() {
                for k in 0..NUM_RULES {
                    // max ratio considers absent examples, actual ratio does not
                    let ratio = self.weak_rules_score[i][j][k] / self.total_weight;
                    if ratio >= max_ratio {
                        max_ratio = ratio;
                        actual_ratio = ratio;
                        rule_id = (i, j, k);
                    }
                }
            }
        }
        (actual_ratio, rule_id)
    }
    */

    pub fn is_gamma_significant(&self) -> bool {
        self.rho_gamma >= self.min_gamma
    }

    /// Update the statistics of all candidate weak rules using current batch of
    /// training examples.
    // validate_set1: &Vec<Example>, validate_w1: &Vec<f32>,
    // validate_set2: &Vec<Example>, validate_w2: &Vec<f32>,
    pub fn update(
        &mut self,
        tree: &Model,
        data: &[ExampleInSampleSet],
    ) -> Option<TreeNode> {
        // update global stats
        self.total_count += data.len();
        self.total_weight      += data.par_iter().map(|t| (t.1).0).sum::<f32>();
        self.total_weight_sq   += data.par_iter().map(|t| ((t.1).0) * ((t.1).0)).sum::<f32>();

        // preprocess examples - Complexity: O(Examples * NumRules)
        let expand_node = self.expand_node;
        let rho_gamma = {
            if expand_node == 0 {
                self.root_gamma
            } else {
                self.rho_gamma
            }
        };
        let data: Vec<(f32, (&Example, RuleStats))> = {
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
        };

        // Calculate total sum of the weights and the number of examples
        let (batch_sum_weight, batch_count): (f32, usize) = {
            data.par_iter()
                .map(|(w, _)| (*w, 1))
                .reduce(|| (0.0, 0), |(a1, a2), (b1, b2)| (a1 + b1, a2 + b2))
        };
        self.sum_weight += batch_sum_weight;
        self.count      += batch_count;

        // Update each weak rule - Complexity: O(Bins * Splits)
        let count = self.count;
        let total_weight = self.total_weight;
        let total_weight_sq = self.total_weight_sq;
        let tree_node = {
            // all bins read data in parallel
            self.bins.par_iter().zip(
                self.weak_rules_score.par_iter_mut()
            ).zip(
                self.sum_c_squared.par_iter_mut()
            ).zip(
                self.num_positive.par_iter_mut()
                    .zip(self.num_negative.par_iter_mut())
                    .zip(self.weight_positive.par_iter_mut())
                    .zip(self.weight_negative.par_iter_mut())
            ).enumerate().map(|(i, zipped_values)| {
                let (((bin, weak_rules_score), sum_c_squared), debug_info) = zipped_values;
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
                        let flip_index = example.feature[i] as usize;
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
                                        feature:        i,
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
            }).find_any(|t| t.is_some()).unwrap_or(None)
        };
        tree_node
    }

    pub fn set_gamma(&mut self, gamma: f32, root_gamma: f32) {
        if !is_zero(gamma - self.rho_gamma) || !is_zero(root_gamma - self.root_gamma) {
            debug!("set-gamma, {}, {}, {}, {}", self.rho_gamma, self.root_gamma, gamma, root_gamma);
            self.rho_gamma = gamma;
            self.root_gamma = root_gamma;
            self.reset();
        }
    }

    pub fn set_expand_node(&mut self, expand_node: usize) -> bool {
        if expand_node != self.expand_node {
            debug!("set-expand-node, {}, {}", self.expand_node, expand_node);
            self.expand_node = expand_node;
            true
        } else {
            false
        }
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