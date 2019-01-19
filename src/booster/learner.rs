use rayon::prelude::*;

use std::collections::HashMap;
use std::f32::INFINITY;
use std::ops::Range;

use commons::ExampleInSampleSet;
use tree::Tree;
use super::super::Example;

use buffer_loader::BufferLoader;
use commons::bins::Bins;
use commons::max;
use commons::min;
use commons::get_bound;
use commons::get_weight;

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
// [i][j][k][0,1] => bin i slot j node k
const GAMMA_GAP: f32 = 0.0;
const NUM_RULES: usize = 2;
type ScoreBoard = Vec<Vec<Vec<[f32; NUM_RULES]>>>;

const RULES: [[f32; 2]; NUM_RULES] = [[1.0, -1.0], [-1.0, 1.0]];


/// A weak rule with an edge larger or equal to the targetting value of `gamma`
struct TreeNode {
    tree_index: usize,
    node_type: usize,
    feature: usize,
    threshold: f32,
    left_predict: f32,
    right_predict: f32,

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
            "tree-node-info, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
            self.tree_index,
            self.node_type,
            self.feature,
            self.threshold,
            self.left_predict,
            self.right_predict,
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
pub struct Learner {
    bins: Vec<Bins>,
    range_start: usize,
    num_examples_before_shrink: usize,

    weak_rules_score: ScoreBoard,
    sum_c:            ScoreBoard,
    sum_c_squared:    ScoreBoard,

    rho_gamma: f32,
    root_rho_gamma: f32,
    tree_max_rho_gamma: f32,
    counts: Vec<usize>,  // track the number of examples exposed to each splitting candidates
    sum_weights: Vec<f32>,
    is_active: Vec<bool>,

    default_gamma: f32,
    min_gamma: f32,
    num_candid: usize,
    pub total_count: usize,
    total_weight: f32,

    tree: Tree,
    max_leaves: usize,
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
        max_leaves: usize,
        min_gamma: f32,
        default_gamma: f32,
        num_examples_before_shrink: u32,
        bins: Vec<Bins>,
        range: &Range<usize>,
    ) -> Learner {
        let mut learner = Learner {
            range_start: range.start,
            num_examples_before_shrink: num_examples_before_shrink as usize,

            weak_rules_score: vec![vec![]; bins.len()],
            sum_c:            vec![vec![]; bins.len()],
            sum_c_squared:    vec![vec![]; bins.len()],

            counts: vec![],
            sum_weights: vec![],
            rho_gamma: default_gamma.clone(),
            root_rho_gamma: default_gamma.clone(),
            tree_max_rho_gamma: 0.0,
            is_active: vec![],

            default_gamma: default_gamma,
            min_gamma: min_gamma,
            num_candid: 0,
            total_count: 0,
            total_weight: 0.0,

            tree: Tree::new((max_leaves * 2 - 1) as u16),
            max_leaves: max_leaves,
            bins: bins,
        };
        learner.setup(0);
        learner
    }

    /// Reset the statistics of all candidate weak rules
    /// (except gamma, because the advantage of the root node is not likely to increase)
    /// Trigger when the model is changed (i.e. the weight distribution of examples)
    pub fn reset_all(&mut self) {
        self.reset_trackers();
        self.is_active.iter_mut().for_each(|t| { *t = false; });
        self.num_candid = 0;
        self.tree_max_rho_gamma = 0.0;
        self.setup(0);
        self.rho_gamma = self.root_rho_gamma;
        debug!("reset-all, {}", self.rho_gamma);
    }

    /// Reset the statistics of the speicified candidate weak rules
    /// Trigger when the gamma is changed and new node is added
    fn reset_trackers(&mut self) {
        for i in 0..self.bins.len() {
            for index in 0..self.num_candid {
                if self.is_active[index] {
                    for j in 0..self.bins[i].len() {
                        for k in 0..NUM_RULES {
                            self.weak_rules_score[i][index][j][k] = 0.0;
                            self.sum_c[i][index][j][k]            = 0.0;
                            self.sum_c_squared[i][index][j][k]    = 0.0;
                        }
                    }
                }
            }
        }
        for index in 0..self.num_candid {
            self.sum_weights[index] = 0.0;
            self.counts[index] = 0;
        }
        self.total_count = 0;
        self.total_weight = 0.0;
    }

    fn setup(&mut self, index: usize) {
        let mut is_cleared = false;
        while index >= self.is_active.len() {
            if self.is_active.len() == index {
                is_cleared = true;
            }
            for i in 0..self.bins.len() {
                let len = self.bins[i].len();
                self.weak_rules_score[i].push(vec![[0.0; NUM_RULES]; len]);
                self.sum_c[i].push(vec![[0.0; NUM_RULES]; len]);
                self.sum_c_squared[i].push(vec![[0.0; NUM_RULES]; len]);
            }
            self.sum_weights.push(0.0);
            self.counts.push(0);
            self.is_active.push(false);
        }
        if !is_cleared {
            for i in 0..self.bins.len() {
                for j in 0..self.bins[i].len() {
                    for k in 0..NUM_RULES {
                        self.weak_rules_score[i][index][j][k] = 0.0;
                        self.sum_c[i][index][j][k]            = 0.0;
                        self.sum_c_squared[i][index][j][k]    = 0.0;
                    }
                }
            }
            self.sum_weights[index] = 0.0;
            self.counts[index] = 0;
        }
        self.is_active[index] = true;
        self.num_candid = max(self.num_candid, index + 1);
        self.rho_gamma = self.default_gamma;
    }

    fn get_max_empirical_ratio(&self) -> (f32, (usize, usize, usize, usize)) {
        let indices: Vec<usize> = self.is_active.iter().enumerate()
                                      .filter(|(_, is_active)| **is_active)
                                      .map(|(index, _)| index)
                                      .collect();
        let mut max_ratio = 0.0;
        let mut actual_ratio = 0.0;
        let mut rule_id = None;
        for i in 0..self.bins.len() {
            for index in indices.iter() {
                for j in 0..self.bins[i].len() {
                    for k in 0..NUM_RULES {
                        // max ratio considers absent examples, actual ratio does not
                        let ratio = self.weak_rules_score[i][*index][j][k] / self.total_weight;
                        if ratio >= max_ratio {
                            max_ratio = ratio;
                            actual_ratio =
                                self.weak_rules_score[i][*index][j][k] / self.sum_weights[*index];
                            rule_id = Some((i, j, *index, k));
                        }
                    }
                }
            }
        }
        (actual_ratio, rule_id.unwrap())
    }

    pub fn is_gamma_significant(&self) -> bool {
        self.tree_max_rho_gamma >= self.min_gamma
    }

    /// Update the statistics of all candidate weak rules using current batch of
    /// training examples. 
    pub fn update(&mut self, data: &[ExampleInSampleSet]) -> Option<Tree> {
        // update global stats
        self.total_count += data.len();
        self.total_weight += data.par_iter().map(|t| (t.1).0).sum::<f32>();

        // preprocess examples - Complexity: O(Examples * NumRules)
        let rho_gamma = self.rho_gamma;
        let data: Vec<(usize, f32, (&Example, [[[f32; 2]; 3]; NUM_RULES]))> = {
            data.par_iter().map(|(example, (weight, _))| {
                let (index, pred) = self.tree.get_leaf_index_prediction(example);
                let weight = weight * get_weight(example, pred);
                let labeled_weight = weight * (example.label as f32);
                let null_weight = 2.0 * rho_gamma * weight;
                let mut vals = [[[0.0; 2]; 3]; NUM_RULES];
                RULES.iter().enumerate().for_each(|(i, pred)| {
                    let left_abs_val  = pred[0] * labeled_weight;
                    let left_ci       = left_abs_val - null_weight;
                    let right_abs_val = pred[1] * labeled_weight;
                    let right_ci      = right_abs_val - null_weight;
                    vals[i][0] = [left_abs_val, right_abs_val];
                    vals[i][1] = [left_ci, right_ci];
                    vals[i][2] = [left_ci * left_ci, right_ci * right_ci];
                });
                (index, weight, (example, vals))
            }).collect()
        };
        // Sort examples - Complexity: O(Examples)
        let mut data_by_node: HashMap<usize, Vec<(&Example, [[[f32; 2]; 3]; NUM_RULES])>> =
            HashMap::new();
        data.into_iter().for_each(|(index, weight, value)| {
            // Update weights and counts
            self.sum_weights[index] += weight;
            self.counts[index] += 1;
            data_by_node.entry(index).or_insert(Vec::new()).push(value);
        });

        // Update each weak rule - Complexity: O(Candid * Bins * Splits)
        let range_start = self.range_start;
        let num_candid = self.num_candid;
        let counts = self.counts.clone();
        let mut valid_tree_node = None;
        for index in 0..num_candid { // Splitting node candidate index
            if !data_by_node.contains_key(&index) {
                continue;
            }
            let data = &data_by_node[&index];
            let tree_node = {
                self.bins.par_iter().zip(
                    self.weak_rules_score.par_iter_mut()
                ).zip(
                    self.sum_c.par_iter_mut()
                ).zip(
                    self.sum_c_squared.par_iter_mut()
                ).enumerate().map(|(i, zipped_values)| {
                    let (((bin, weak_rules_score), sum_c), sum_c_squared) = zipped_values;

                    // <Split, NodeId, RuleId, stats, LeftOrRight>
                    // the last element of is for the examples that are larger than all split values
                    let mut bin_accum_vals =
                        vec![vec![[[0.0; 2]; 3]; NUM_RULES]; bin.len() + 1];
                    data.iter()
                        .for_each(|(example, vals)| {
                            // complexity: O(log N)
                            let flip_index = example.feature[range_start + i] as usize;
                            let accums = &mut bin_accum_vals[flip_index];
                            for j in 0..NUM_RULES {
                                for k in 0..3 {  // 3 trackers
                                    accums[j][k][0] += vals[j][k][0];
                                    accums[j][k][1] += vals[j][k][1];
                                }
                            }
                        });

                    let mut accum_left  = vec![[0.0; 3]; NUM_RULES];
                    let mut accum_right = vec![[0.0; 3]; NUM_RULES];
                    // Accumulate sum of the stats of all examples that go to the right child
                    for j in 0..bin.len() { // Split value
                        for rule_idx in 0..NUM_RULES { // Types of rule
                            for it in 0..3 {  // 3 trackers
                                accum_right[rule_idx][it] +=
                                    bin_accum_vals[j][rule_idx][it][1];
                            }
                        }
                    }
                    // Now update each splitting values of the bin
                    let mut valid_weak_rule = None;
                    bin.get_vals().iter().enumerate().for_each(|(j, threshold)| {
                        for rule_idx in 0..NUM_RULES { // Types of rule
                            for it in 0..3 { // Move examples from the right to the left child
                                accum_left[rule_idx][it]  +=
                                    bin_accum_vals[j][rule_idx][it][0];
                                accum_right[rule_idx][it] -=
                                    bin_accum_vals[j][rule_idx][it][1];
                            }

                            let weak_rules_score = &mut weak_rules_score[index][j][rule_idx];
                            let sum_c            = &mut sum_c[index][j][rule_idx];
                            let sum_c_squared    = &mut sum_c_squared[index][j][rule_idx];
                            *weak_rules_score += accum_left[rule_idx][0] + accum_right[rule_idx][0];
                            *sum_c            += accum_left[rule_idx][1] + accum_right[rule_idx][1];
                            *sum_c_squared    += accum_left[rule_idx][2] + accum_right[rule_idx][2];
                            // Check stopping rule
                            let count = counts[index];
                            let bound = get_bound(count, *sum_c, *sum_c_squared);
                            let bound = bound.unwrap_or(INFINITY);
                            if *sum_c > bound {
                                let base_pred = 0.5 * (
                                    (0.5 + rho_gamma + GAMMA_GAP) / (0.5 - rho_gamma - GAMMA_GAP)
                                ).ln();
                                valid_weak_rule = Some(
                                    TreeNode {
                                        tree_index:     index,
                                        node_type:      rule_idx,
                                        feature:        i + range_start,
                                        threshold:      *threshold,
                                        left_predict:   base_pred * RULES[rule_idx][0],
                                        right_predict:  base_pred * RULES[rule_idx][1],
                                        gamma:          rho_gamma,
                                        raw_martingale: *weak_rules_score,
                                        sum_c:          *sum_c,
                                        sum_c_squared:  *sum_c_squared,
                                        bound:          bound,
                                        num_scanned:    count,
                                        fallback:       false,
                                    }
                                );
                            }
                        }
                    });
                    valid_weak_rule
                }).find_any(|t| t.is_some()).unwrap_or(None)
            };
            if valid_tree_node.is_none() && tree_node.is_some() {
                valid_tree_node = tree_node;
                break;
            }
        }

        let tree_node =
            if valid_tree_node.is_some() || self.total_count <= self.num_examples_before_shrink {
                valid_tree_node
            } else {
                // cannot find a valid weak rule, need to fallback and shrink gamma
                let old_rho_gamma = self.rho_gamma;
                let (empirical_gamma, (i, j, index, k)) = self.get_max_empirical_ratio();
                let empirical_gamma = empirical_gamma / 2.0;
                let bounded_empirical_gamma = min(0.25, empirical_gamma);
                // Fallback prepare
                let base_pred =
                    0.5 * ((0.5 + bounded_empirical_gamma) / (0.5 - bounded_empirical_gamma)).ln();
                let count          = self.counts[index];
                let raw_martingale = self.weak_rules_score[i][index][j][k];
                let sum_c          = self.sum_c[i][index][j][k];
                let sum_c_squared  = self.sum_c_squared[i][index][j][k];
                let bound = get_bound(count, sum_c, sum_c_squared).unwrap_or(INFINITY);
                // shrink rho_gamma
                // self.rho_gamma = 0.9 * min(old_rho_gamma, empirical_gamma);
                if self.is_active[0] {
                    self.root_rho_gamma = self.root_rho_gamma * 0.9;
                }
                // trackers will reset later
                // debug!("shrink-gamma, {}, {}, {}",
                //         old_rho_gamma, empirical_gamma, self.rho_gamma);
                // generate a fallback tree node
                Some(TreeNode {
                    tree_index:     index,
                    node_type:      k,
                    feature:        i + range_start,
                    threshold:      self.bins[i].get_vals()[j],
                    left_predict:   base_pred * RULES[k][0],
                    right_predict:  base_pred * RULES[k][1],

                    gamma:          empirical_gamma,
                    raw_martingale: raw_martingale,
                    sum_c:          sum_c,
                    sum_c_squared:  sum_c_squared,
                    bound:          bound,
                    num_scanned:    count,
                    fallback:       true,
                })
            };

        let mut ret = None;
        if tree_node.is_some() {
            let tree_node = tree_node.unwrap();
            tree_node.write_log();
            let (left_node, right_node) = self.tree.split(
                tree_node.tree_index, tree_node.feature, tree_node.threshold,
                tree_node.left_predict, tree_node.right_predict,
            );
            self.is_active[tree_node.tree_index] = false;
            let tree_gamma = tree_node.gamma * {
                if tree_node.fallback { 0.9 } else { 1.0 }
            };
            self.tree_max_rho_gamma = max(self.tree_max_rho_gamma, tree_gamma);
            self.reset_trackers();
            if self.tree.num_leaves == self.max_leaves * 2 - 1 {
                debug!("default-gamma, {}, {}", self.default_gamma, self.tree_max_rho_gamma * 0.9);
                self.default_gamma = self.tree_max_rho_gamma;
                // A new tree is created
                self.tree.release();
                ret = Some(self.tree.clone());
                self.tree = Tree::new((self.max_leaves * 2 - 1) as u16);
            } else {
                // Tracking weak rules on the new candidate leaves
                self.setup(left_node as usize);
                self.setup(right_node as usize);
            }
        }
        ret
    }
}


pub fn get_base_tree(max_sample_size: usize, data_loader: &mut BufferLoader) -> (Tree, f32) {
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
    let mut tree = Tree::new(2);
    tree.split(0, 0, 0.0, prediction, prediction);
    tree.release();

    info!("root-tree-info, {}, {}, {}, {}", 1, max_sample_size, gamma, gamma * gamma);
    (tree, gamma)
}
