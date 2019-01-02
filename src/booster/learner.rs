use rayon::prelude::*;

use std::f32::INFINITY;
use std::ops::Range;

use commons::ExampleInSampleSet;
use tree::Tree;
use super::bins::Bins;
use super::super::Example;

use buffer_loader::BufferLoader;
use commons::is_zero;
use commons::min;
use commons::get_bound;
use commons::get_relative_weights;
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
type ScoreBoard = Vec<Vec<Vec<[f32; 2]>>>;

const LEFT_NODE:  [f32; 2] = [1.0, -1.0];
const RIGHT_NODE: [f32; 2] = [-1.0, 1.0];

/// A weak rule with an edge larger or equal to the targetting value of `gamma`
struct TreeNode {
    tree_index: usize,
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
            "tree-node-info, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
            self.tree_index,
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

            weak_rules_score: bins.iter().map(|bin| vec![vec![]; bin.len()]).collect(),
            sum_c:            bins.iter().map(|bin| vec![vec![]; bin.len()]).collect(),
            sum_c_squared:    bins.iter().map(|bin| vec![vec![]; bin.len()]).collect(),

            counts: vec![],
            sum_weights: vec![],
            rho_gamma: default_gamma.clone(),
            root_rho_gamma: default_gamma.clone(),
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
        for i in 0..self.bins.len() {
            for j in 0..self.bins[i].len() {
                self.weak_rules_score[i][j].clear();
                self.sum_c[i][j].clear();
                self.sum_c_squared[i][j].clear();
            }
        }
        self.sum_weights.clear();
        self.counts.clear();
        self.is_active.clear();
        self.total_count = 0;
        self.total_weight = 0.0;
        self.num_candid = 0;
        self.setup(0);
        self.rho_gamma = self.root_rho_gamma;
    }

    /// Reset the statistics of the speicified candidate weak rules
    /// Trigger when the gamma is changed
    fn reset_shrink(&mut self) {
        for i in 0..self.bins.len() {
            for j in 0..self.bins[i].len() {
                for index in 0..self.num_candid {
                    if self.is_active[index] {
                        for k in 0..2 {
                            self.weak_rules_score[i][j][index][k] = 0.0;
                            self.sum_c[i][j][index][k]            = 0.0;
                            self.sum_c_squared[i][j][index][k]    = 0.0;
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
        let b = [0.0; 2];
        while index >= self.num_candid {
            for i in 0..self.bins.len() {
                for j in 0..self.bins[i].len() {
                    self.weak_rules_score[i][j].push(b);
                    self.sum_c[i][j].push(b);
                    self.sum_c_squared[i][j].push(b);
                }
            }
            self.sum_weights.push(0.0);
            self.counts.push(0);
            self.is_active.push(false);
            self.num_candid += 1;
        }
        self.is_active[index] = true;
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
            for j in 0..self.bins[i].len() {
                for index in indices.iter() {
                    for k in 0..2 {
                        // max ratio considers absent examples, actual ratio does not
                        let ratio = self.weak_rules_score[i][j][*index][k] / self.total_weight;
                        if ratio >= max_ratio {
                            max_ratio = ratio;
                            actual_ratio =
                                self.weak_rules_score[i][j][*index][k] / self.sum_weights[*index];
                            rule_id = Some((i, j, *index, k));
                        }
                    }
                }
            }
        }
        (actual_ratio, rule_id.unwrap())
    }

    pub fn is_gamma_significant(&self) -> bool {
        self.rho_gamma >= self.min_gamma
    }

    /// Update the statistics of all candidate weak rules using current batch of
    /// training examples. 
    pub fn update(&mut self, data: &[ExampleInSampleSet]) -> Option<Tree> {
        // update global stats
        let weights = get_relative_weights(data);
        self.total_count += data.len();
        self.total_weight += weights.iter().sum::<f32>();

        // preprocess examples
        let rho_gamma = self.rho_gamma;
        let min_gamma = self.min_gamma;
        let data = {
            let mut data: Vec<(usize, f32, &Example, ([f32; 2], [f32; 2]), f32, f32)> =
                data.par_iter().zip(weights.par_iter()).map(|(example, weight)| {
                    let example = &example.0;
                    let (index, pred) = self.tree.get_leaf_index_prediction(example);
                    let weight = weight * get_weight(example, pred);
                    if rho_gamma >= min_gamma {
                        let labeled_weight = weight * (example.label as f32);
                        let null_weight = 2.0 * rho_gamma * weight;
                        let c_sq = ((1.0 + 2.0 * rho_gamma) * weight).powi(2);
                        let left_score: Vec<_> =
                            LEFT_NODE.iter().map(|sign| sign * labeled_weight).collect();
                        let right_score: Vec<_> =
                            RIGHT_NODE.iter().map(|sign| sign * labeled_weight).collect();
                        let s_labeled_weight = (
                            [left_score[0], left_score[1]], [right_score[0], right_score[1]],
                        );
                        Some((index, weight, example, s_labeled_weight, null_weight, c_sq))
                    } else {
                        None
                    }
                }).filter(|t| t.is_some()).map(|t| t.unwrap()).collect();
            data.sort_by(|a, b| (a.0).cmp(&b.0));
            data
        };

        // Update weights and counts
        data.iter().for_each(|(index, weight, _, _, _, _)| {
            self.sum_weights[*index] += weight;
            self.counts[*index] += 1;
        });

        // update each weak rule
        let is_active = self.is_active.clone();
        let counts = self.counts.clone();
        let range_start = self.range_start;
        let num_candid = self.num_candid;
        let valid_tree_node = self.bins.par_iter().zip(
            self.weak_rules_score.par_iter_mut()
        ).zip(
            self.sum_c.par_iter_mut()
        ).zip(
            self.sum_c_squared.par_iter_mut()
        ).enumerate().map(|(i, (((bin, weak_rules_score), sum_c), sum_c_squared))| {
            // Update stats
            data.iter().for_each(|(index, _, example, labeled_weight, null_weight, c_sq)| {
                let index = *index;
                let feature_val = (*example).feature[i + range_start] as f32;
                bin.get_vals().iter().enumerate().for_each(|(j, threshold)| {
                    let labeled_weight =
                        if feature_val <= *threshold {
                            &(*labeled_weight).0
                        } else {
                            &(*labeled_weight).1
                        };

                    weak_rules_score[j][index][0] += labeled_weight[0];
                    weak_rules_score[j][index][1] += labeled_weight[1];

                    sum_c[j][index][0]            += labeled_weight[0] - null_weight;
                    sum_c[j][index][1]            += labeled_weight[1] - null_weight;

                    sum_c_squared[j][index][0]    += (1.0 + 2.0 * rho_gamma).powi(2) * c_sq;
                    sum_c_squared[j][index][1]    += (1.0 + 2.0 * rho_gamma).powi(2) * c_sq;
                });
            });

            // check stopping rule
            let mut valid_weak_rule = None;
            bin.get_vals().iter().enumerate().for_each(|(j, threshold)| {
                for index in 0..num_candid {
                    if !is_active[index] || is_zero(sum_c_squared[j][index][0]) {
                        // this candidate received no data or is already generated
                        continue;
                    }
                    let count = counts[index];
                    for k in 0..2 {
                        let weak_rules_score = weak_rules_score[j][index][k];
                        let sum_c            = sum_c[j][index][k];
                        let sum_c_squared    = sum_c_squared[j][index][k];
                        let bound = get_bound(count, sum_c, sum_c_squared).unwrap_or(INFINITY);
                        if sum_c > bound {
                            let base_pred = 0.5 * ((0.5 + rho_gamma) / (0.5 - rho_gamma)).ln();
                            valid_weak_rule = Some(
                                TreeNode {
                                    tree_index:     index,
                                    feature:        i + range_start,
                                    threshold:      *threshold,
                                    left_predict:   base_pred * LEFT_NODE[k],
                                    right_predict:  base_pred * RIGHT_NODE[k],

                                    gamma:          rho_gamma,
                                    raw_martingale: weak_rules_score,
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
        }).find_any(|t| t.is_some()).unwrap_or(None);

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
                let raw_martingale = self.weak_rules_score[i][j][index][k];
                let sum_c          = self.sum_c[i][j][index][k];
                let sum_c_squared  = self.sum_c_squared[i][j][index][k];
                let bound = get_bound(count, sum_c, sum_c_squared).unwrap_or(INFINITY);
                // shrink rho_gamma
                self.rho_gamma = 0.9 * min(old_rho_gamma, empirical_gamma);
                if self.is_active[0] {
                    self.root_rho_gamma = self.rho_gamma;
                }
                self.reset_shrink();
                debug!("shrink-gamma, {}, {}, {}",
                        old_rho_gamma, empirical_gamma, self.rho_gamma);
                // generate a fallback tree node
                Some(TreeNode {
                    tree_index:     index,
                    feature:        i + range_start,
                    threshold:      self.bins[i].get_vals()[j],
                    left_predict:   base_pred * LEFT_NODE[k],
                    right_predict:  base_pred * RIGHT_NODE[k],

                    gamma:          old_rho_gamma,
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
            self.total_count = 0;
            self.total_weight = 0.0;
            if self.tree.num_leaves == self.max_leaves * 2 - 1 {
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
                |(num_pos, num_neg), (example, _, _)| {
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
