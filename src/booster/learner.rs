use rayon::prelude::*;

use std::collections::HashMap;
use std::ops::Range;

use commons::ExampleInSampleSet;
use super::super::Example;
use super::super::TFeature;

use buffer_loader::BufferLoader;
use commons::bins::Bins;
use commons::min;
use commons::get_bound;
use commons::Model;

// TODO: The tree generation and score updates are for AdaBoost only,
// extend it to other potential functions

/*
TODO: extend support to regression tasks
TODO: re-use ScoreBoard space of the generated rules to reduce the memory footprint by half
      (just adding a mapping from the index here to the index in the tree should do the job)

Structure:
    Feature index -> split value index -> candidate splitting node index -> prediction types

Each split corresponds to 2 types of predictions,
    1. Left +1, Right -1;
    2. Left -1, Right +1;
*/
const NUM_PREDS: usize = 2;
const NUM_RULES: usize = 2 * NUM_PREDS;  // i.e., false (0), true (1)
const PREDS: [f32; NUM_PREDS] = [-1.0, 1.0];
// (pred1, false), (pred1, true), (pred2, false), (pred2, true)
type ScoreBoard = Vec<Vec<Vec<[f32; NUM_RULES]>>>;
type RuleStats = [[f32; 2]; NUM_PREDS];


/// A weak rule with an edge larger or equal to the targetting value of `gamma`
pub struct TreeNode {
    pub prt_index: usize,
    pub feature: usize,
    pub threshold: TFeature,
    pub evaluation: bool,
    pub predict: f32,

    pub gamma: f32,
    raw_martingale: f32,
    sum_c: f32,
    sum_c_squared: f32,
    bound: f32,
    num_scanned: usize,
    pub fallback: bool,
}

impl TreeNode {
    pub fn write_log(&self) {
        info!(
            "tree-node-info, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
            self.prt_index,
            self.feature,
            self.threshold,
            self.evaluation,
            self.predict,

            self.num_scanned,
            self.gamma,
            self.raw_martingale,
            self.sum_c,
            self.sum_c_squared,
            self.bound,
            self.sum_c_squared / self.num_scanned as f32,
            self.fallback,
        );
    }
}


/// Statisitics of all weak rules that are being evaluated.
/// The objective of `Learner` is to find a weak rule that satisfies the condition of
/// the stopping rule.
// [i][k][j][rule_id] => bin i slot j candidate-node k
pub struct Learner {
    bins: Vec<Bins>,
    range_start: usize,
    num_candid:  usize,
    num_examples_before_shrink: usize,
    _default_gamma: f32,
    min_gamma: f32,

    rho_gamma:        f32,
    alist:            Vec<usize>,
    alist_tail:       usize,
    // global trackers
    pub total_count:  usize,
    total_weight:     f32,
    total_weight_sq:  f32,
    // trackers for each candidate
    weak_rules_score: ScoreBoard,
    sum_c_squared:    ScoreBoard,
    // track the number of examples exposed to each splitting candidates
    counts:           HashMap<usize, usize>,
    sum_weights:      HashMap<usize, f32>,
}

impl Learner {
    /// Create a `Learner` that search for valid weak rules.
    /// `default_gamma` is the initial value of the edge `gamma`.
    /// `bins` is vectors of the all thresholds on all candidate features for generating weak rules.
    ///
    /// `range` is the range of candidate features for generating weak rules. In most cases,
    /// if the algorithm is running on a single worker, `range` is 0..`num_of_features`;
    /// if the algorithm is running on multiple workers, `range` is a subset of the feature set.
    pub fn new(
        min_gamma: f32,
        default_gamma: f32,
        num_examples_before_shrink: u32,
        num_candid: usize,
        bins: Vec<Bins>,
        range: &Range<usize>,
    ) -> Learner {
        let mut learner = Learner {
            bins: bins,
            range_start: range.start,
            num_candid:  num_candid.clone(),
            num_examples_before_shrink: num_examples_before_shrink as usize,
            _default_gamma: default_gamma.clone(),
            min_gamma: min_gamma,

            rho_gamma:        default_gamma.clone(),
            alist:            Vec::with_capacity(num_candid),
            alist_tail:       0,
            total_count:      0,
            total_weight:     0.0,
            total_weight_sq:  0.0,
            weak_rules_score: vec![],
            sum_c_squared:    vec![],
            counts:           HashMap::new(),
            sum_weights:      HashMap::new(),
        };
        let num_candid = learner.num_candid;
        for i in 0..learner.bins.len() {
            let size = learner.bins[i].len();
            learner.weak_rules_score.push(vec![vec![[0.0; NUM_RULES]; size]; num_candid]);
            learner.sum_c_squared.push(vec![vec![[0.0; NUM_RULES]; size]; num_candid]);
        }
        learner
    }

    /// Reset the statistics of all candidate weak rules
    /// (except gamma, because the advantage of the root node is not likely to increase)
    /// Trigger when the model or the gamma is changed
    pub fn reset(&mut self) {
        for i in 0..self.bins.len() {
            for index in 0..self.num_candid {
                for j in 0..self.bins[i].len() {
                    for k in 0..NUM_RULES {
                        self.weak_rules_score[i][index][j][k] = 0.0;
                        self.sum_c_squared[i][index][j][k]    = 0.0;
                    }
                }
            }
        }
        self.sum_weights.clear();
        self.counts.clear();
        self.total_count = 0;
        self.total_weight = 0.0;
        self.total_weight_sq = 0.0;
    }

    fn setup(&mut self, index: usize, c_idx: usize) {
        for i in 0..self.bins.len() {
            for j in 0..self.bins[i].len() {
                for k in 0..NUM_RULES {
                    self.weak_rules_score[i][c_idx][j][k] = 0.0;
                    self.sum_c_squared[i][c_idx][j][k]    = 0.0;
                }
            }
        }
        *self.sum_weights.entry(index).or_insert(0.0) = 0.0;
        *self.counts.entry(index).or_insert(0) = 0;
        debug!("setup, {}, {}", index, c_idx);
    }

    fn get_max_empirical_ratio(&self) -> (f32, (usize, usize, usize, usize, usize)) {
        let mut max_ratio = 0.0;
        let mut actual_ratio = 0.0;
        let mut rule_id = (0, 0, 0, 0, 0);
        for i in 0..self.bins.len() {
            self.alist.iter().enumerate().for_each(|(c_idx, index)| {
                for j in 0..self.bins[i].len() {
                    for k in 0..NUM_RULES {
                        // max ratio considers absent examples, actual ratio does not
                        let ratio = self.weak_rules_score[i][c_idx][j][k] / self.total_weight;
                        if ratio >= max_ratio {
                            max_ratio = ratio;
                            actual_ratio = ratio;
                            //    self.weak_rules_score[i][c_idx][j][k] / self.sum_weights[index];
                            rule_id = (i, j, c_idx, *index, k);
                        }
                    }
                }
            });
        }
        (actual_ratio, rule_id)
    }

    pub fn is_gamma_significant(&self) -> bool {
        self.rho_gamma >= self.min_gamma
    }

    // TODO: support other activation function
    pub fn push_active(&mut self, index: usize) -> Option<usize> {
        let mut deactive = None;
        let c_idx = {
            if self.alist.len() < self.num_candid {
                self.alist.push(index);
                self.alist.len() - 1
            } else {
                let c_idx = self.alist_tail;
                self.alist_tail = (self.alist_tail + 1) % self.num_candid;
                deactive = Some(self.alist[c_idx]);
                self.alist[c_idx] = index;
                c_idx
            }
        };
        debug!("push-active, {}, {:?}, {}", c_idx, deactive, index);
        self.setup(index, c_idx);
        deactive
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
        let rho_gamma = self.rho_gamma;
        let data: Vec<(Vec<usize>, f32, (&Example, RuleStats))> = {
            data.par_iter().map(|(example, (weight, _))| {
                let labeled_weight = weight * (example.label as f32);
                // let null_weight = 2.0 * rho_gamma * weight;
                let null_weight = 2.0 * rho_gamma * weight;
                let mut vals: RuleStats = [[0.0; 2]; NUM_PREDS];
                PREDS.iter().enumerate().for_each(|(i, pred)| {
                    let abs_val = pred * labeled_weight;
                    let ci      = abs_val - null_weight;
                    vals[i][0]  = abs_val;
                    vals[i][1]  = ci * ci - null_weight * null_weight;
                });
                (tree.get_active_nodes(example), *weight, (example, vals))
            }).collect()
        };
        // Sort examples - Complexity: O(Examples)
        let mut data_by_node: HashMap<usize, Vec<&(&Example, RuleStats)>> = HashMap::new();
        data.iter().for_each(|(indices, weight, value)| {
            // Update weights and counts
            indices.iter().for_each(|index| {
                *self.sum_weights.entry(*index).or_insert(0.0) += *weight;
                *self.counts.entry(*index).or_insert(0) += 1;
                data_by_node.entry(*index).or_insert(Vec::new()).push(value);
            });
        });

        // Update each weak rule - Complexity: O(Candid * Bins * Splits)
        let range_start = self.range_start;
        let counts = self.counts.clone();
        let total_weight = self.total_weight;
        let total_weight_sq = self.total_weight_sq;
        let mut valid_tree_node = None;
        let alist = self.alist.clone();
        alist.iter().enumerate()
             .filter(|(_, index)| data_by_node.contains_key(index))
             .for_each(|(c_idx, index)| {
                 let data = &data_by_node[index];
                 let tree_node = {
                     // all bins read data in parallel
                     self.bins.par_iter().zip(
                         self.weak_rules_score.par_iter_mut()
                     ).zip(
                         self.sum_c_squared.par_iter_mut()
                     ).enumerate().map(|(i, zipped_values)| {
                         let ((bin, weak_rules_score), sum_c_squared) = zipped_values;

                         // <Split, NodeId, RuleId, stats, LeftOrRight>
                         // the last element of is for the examples that are larger than all split values
                         let mut bin_accum_vals: Vec<RuleStats> =
                             vec![[[0.0; 2]; NUM_PREDS]; bin.len() + 1];
                         data.iter()
                             .for_each(|(example, vals)| {
                                 let flip_index = example.feature[range_start + i] as usize;
                                 let t = &mut bin_accum_vals[flip_index];
                                 for j in 0..NUM_PREDS {
                                     for k in 0..t[j].len() {
                                         t[j][k] += vals[j][k];
                                     }
                                 }
                             });

                         let mut accum_left:  RuleStats = [[0.0; 2]; NUM_PREDS];
                         let mut accum_right: RuleStats = [[0.0; 2]; NUM_PREDS];
                         // Accumulate sum of the stats of all examples that go to the right child
                         for j in 0..bin.len() { // Split value
                             for pred_idx in 0..NUM_PREDS { // Types of rule
                                 for it in 0..accum_right[pred_idx].len() {
                                     accum_right[pred_idx][it] +=
                                         bin_accum_vals[j][pred_idx][it];
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
                                         bin_accum_vals[j][pred_idx][it];
                                     accum_right[pred_idx][it] -=
                                         bin_accum_vals[j][pred_idx][it];
                                 }

                                 // eval == False (right), eval == True (left)
                                 let accums = [&accum_right[pred_idx], &accum_left[pred_idx]];
                                 for (eval_idx, accum) in accums.iter().enumerate() {
                                     let rule_idx = pred_idx * 2 + eval_idx;
                                     let weak_rules_score =
                                            &mut weak_rules_score[c_idx][j][rule_idx];
                                     let sum_c_squared    = &mut sum_c_squared[c_idx][j][rule_idx];
                                     *weak_rules_score   += accum[0];
                                     *sum_c_squared      += accum[1];
                                     // Check stopping rule
                                     let count = counts[index];
                                     let sum_c = *weak_rules_score - 2.0 * rho_gamma * total_weight;
                                     let sum_c_squared = *sum_c_squared +
                                            4.0 * rho_gamma * rho_gamma * total_weight_sq;
                                     let bound = get_bound(sum_c, sum_c_squared);
                                     if sum_c > bound {
                                         let base_pred = 0.5 * (
                                             (0.5 + rho_gamma) / (0.5 - rho_gamma)
                                         ).ln();
                                         valid_weak_rule = Some(
                                             TreeNode {
                                                 prt_index:     *index,
                                                 feature:        i + range_start,
                                                 threshold:      j as TFeature,
                                                 evaluation:     eval_idx == 1,
                                                 predict:        base_pred * PREDS[pred_idx],

                                                 gamma:          rho_gamma,
                                                 raw_martingale: *weak_rules_score,
                                                 sum_c:          sum_c,
                                                 sum_c_squared:  sum_c_squared,
                                                 bound:          bound,
                                                 num_scanned:    count,
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
                 if valid_tree_node.is_none() && tree_node.is_some() {
                     valid_tree_node = tree_node;
                 }
             });

        if valid_tree_node.is_some() || self.total_count <= self.num_examples_before_shrink {
            valid_tree_node
        } else {
            // cannot find a valid weak rule, need to fallback and shrink gamma
            let (empirical_gamma, (i, j, c_idx, index, k)) = self.get_max_empirical_ratio();
            debug!("fallback, {}, {}, {}, {}, {}, {}", empirical_gamma, i, j, c_idx, index, k);
            let (pred_idx, eval_idx) = (k / 2, k % 2);
            let empirical_gamma = empirical_gamma / 2.0;
            let bounded_empirical_gamma = min(0.25, empirical_gamma);
            // Fallback prepare
            let base_pred =
                0.5 * ((0.5 + bounded_empirical_gamma) / (0.5 - bounded_empirical_gamma)).ln();
            let count          = self.counts[&index];
            let raw_martingale = self.weak_rules_score[i][c_idx][j][k];
            let sum_c_squared  = self.sum_c_squared[i][c_idx][j][k];
            let sum_c          = raw_martingale - 2.0 * self.rho_gamma * self.total_weight;
            let sum_c_squared  = sum_c_squared +
                    4.0 * self.rho_gamma * self.rho_gamma * self.total_weight_sq;
            let bound          = get_bound(sum_c, sum_c_squared);
            // shrink rho_gamma	
            let old_rho_gamma = self.rho_gamma;	
            self.rho_gamma = 0.9 * min(old_rho_gamma, empirical_gamma);
            debug!("shrink-gamma, {}, {}, {}", old_rho_gamma, empirical_gamma, self.rho_gamma);
            // generate a fallback tree node
            Some(TreeNode {
                prt_index:     index,
                feature:        i + range_start,
                threshold:      j as TFeature,
                evaluation:     eval_idx == 1,
                predict:        base_pred * PREDS[pred_idx],

                gamma:          empirical_gamma,
                raw_martingale: raw_martingale,
                sum_c:          sum_c,
                sum_c_squared:  sum_c_squared,
                bound:          bound,
                num_scanned:    count,
                fallback:       true,
            })
        }
    }
}


pub fn get_base_node(max_sample_size: usize, data_loader: &mut BufferLoader) -> (f32, f32) {
    let mut sample_size = max_sample_size;
    let mut n_pos = 0;
    let mut n_neg = 0;
    while sample_size > 0 {
        let data = data_loader.get_next_batch(true);
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
    (gamma, prediction)
}


// TODO: clean up the sanity check code
/*
if validate_set1.len() > 0 {
    let (mart1, weight1): (f32, f32) = {
        validate_set1.par_iter().zip(
            validate_w1.par_iter()
        ).map(|(example, w)| {
            let (index, pred) = self.tree.get_leaf_index_prediction(example);
            if index != tree_node.tree_index {
                (0.0, 0.0)
            } else {
                let weight = w * get_weight(example, pred);
                let labeled_weight = weight * (example.label as f32);
                let mart = {
                    if example.feature[tree_node.feature] <= tree_node.threshold {
                        RULES[tree_node.node_type][0] * labeled_weight
                    } else {
                        RULES[tree_node.node_type][1] * labeled_weight
                    }
                };
                (mart, weight)
            }
        }).reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
    };
    let (mart2, weight2): (f32, f32) = {
        validate_set2.par_iter().zip(
            validate_w2.par_iter()
        ).map(|(example, w)| {
            let (index, pred) = self.tree.get_leaf_index_prediction(example);
            if index != tree_node.tree_index {
                (0.0, 0.0)
            } else {
                let weight = w * get_weight(example, pred);
                let labeled_weight = weight * (example.label as f32);
                let mart = {
                    if example.feature[tree_node.feature] <= tree_node.threshold {
                        RULES[tree_node.node_type][0] * labeled_weight
                    } else {
                        RULES[tree_node.node_type][1] * labeled_weight
                    }
                };
                (mart, weight)
            }
        }).reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
    };
    debug!("validate, {}, {}, {}, {}, {}, {}",
        tree_node.fallback, tree_node.num_scanned,
        tree_node.tree_index, tree_node.gamma,
        mart1 / weight1 / 2.0, mart2 / weight2 / 2.0);
}
*/

/*
{
    let tree_node = tree_node.unwrap();
    tree_node.write_log();

    let new_node = self.tree.split(
        tree_node.tree_index, tree_node.feature, tree_node.threshold,
        tree_node.pred_value,
    );
    self.is_active[tree_node.tree_index] = false;
    if tree_node.tree_index > 0 {
        // This is not the root node
        let tree_gamma = tree_node.gamma * {
            if tree_node.fallback { 0.9 } else { 1.0 }
        };
        self.tree_max_rho_gamma = max(self.tree_max_rho_gamma, tree_gamma);
    }
    self.reset_trackers();
    if self.tree.num_leaves == self.max_leaves * 2 - 1 {
        debug!("default-gamma, {}, {}", self.default_gamma, self.tree_max_rho_gamma * 0.9);
        // self.default_gamma = 0.25;
        self.default_gamma = self.tree_max_rho_gamma * 0.9;
        // A new tree is created
        self.tree.release();
        ret = Some(self.tree.clone());
        self.tree = Tree::new((self.max_leaves * 2 - 1) as u16);
    } else {
        // Tracking weak rules on the new candidate leaves
        self.setup(new_node as usize);
    }
}
*/
