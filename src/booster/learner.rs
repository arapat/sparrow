use rayon::prelude::*;

use std::f32::INFINITY;
use std::ops::Range;

use commons::ExampleInSampleSet;
use tree::Tree;
use super::bins::Bins;
use super::super::Example;

use buffer_loader::BufferLoader;
use commons::is_zero;
use commons::max;
use commons::min;
use commons::get_bound;
use commons::get_relative_weights;

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
}

impl TreeNode {
    pub fn write_log(&self) {
        info!(
            "tree-node-info, {}, {}, {}, {}, {}, {}",
            self.num_scanned,
            self.gamma,
            self.raw_martingale,
            self.sum_c,
            self.sum_c_squared,
            self.bound,
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

    rho_gamma: Vec<f32>,  // track the values of gamma on each splitting candidate
    counts: Vec<usize>,  // track the number of examples exposed to each splitting candidates
    sum_weights: Vec<f32>,
    is_active: Vec<bool>,

    default_gamma: f32,
    min_gamma: f32,
    num_candid: usize,
    pub total_count: usize,

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
            rho_gamma: vec![],
            is_active: vec![],

            default_gamma: default_gamma,
            min_gamma: min_gamma,
            num_candid: 0,
            total_count: 0,

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
        for i in 0..self.weak_rules_score.len() {
            for j in 0..self.weak_rules_score[i].len() {
                for index in 0..self.weak_rules_score[i][j].len() {
                    for k in 0..2 {
                        self.weak_rules_score[i][j][index][k] = 0.0;
                        self.sum_c[i][j][index][k]            = 0.0;
                        self.sum_c_squared[i][j][index][k]    = 0.0;
                    }
                }
            }
        }
        for index in 0..self.sum_weights.len() {
            self.sum_weights[index] = 0.0;
            self.counts[index] = 0;
            self.is_active[index] = false;
        }

        self.total_count = 0;
        self.num_candid = 1;
        self.is_active[0] = true;
    }

    /// Reset the statistics of the speicified candidate weak rules
    /// Trigger when the gamma is changed
    fn reset(&mut self, index: usize, rho_gamma: f32) {
        for i in 0..self.weak_rules_score.len() {
            for j in 0..self.weak_rules_score[i].len() {
                for k in 0..2 {
                    self.weak_rules_score[i][j][index][k] = 0.0;
                    self.sum_c[i][j][index][k]            = 0.0;
                    self.sum_c_squared[i][j][index][k]    = 0.0;
                }
            }
        }

        self.sum_weights[index] = 0.0;
        self.counts[index] = 0;
        self.rho_gamma[index] = rho_gamma;
    }

    fn setup(&mut self, index: usize) {
        let b = [0.0; 2];
        while index >= self.sum_weights.len() {
            for i in 0..self.bins.len() {
                let bin = &self.bins[i];
                for j in 0..bin.len() {
                    self.weak_rules_score[i][j].push(b);
                    self.sum_c[i][j].push(b);
                    self.sum_c_squared[i][j].push(b);
                }
            }
            self.sum_weights.push(0.0);
            self.counts.push(0);
            self.rho_gamma.push(0.0);
            self.is_active.push(false);
        }
        self.num_candid = max(self.num_candid, index + 1);
        self.rho_gamma[index] = self.default_gamma;
        self.is_active[index] = true;
    }

    fn get_max_empirical_ratio(&self, index: usize) -> f32 {
        self.weak_rules_score.iter().flat_map(|rules| {
            rules.iter().flat_map(|candidate| {
                candidate[index].iter().map(|s| s / self.sum_weights[index])
            })
        }).fold(0.0, max)
    }

    pub fn is_any_candidate_active(&self) -> bool {
        for i in 0..self.num_candid {
            if self.is_active[i] && self.rho_gamma[i] >= self.min_gamma {
                return true;
            }
        }
        false
    }

    /// Update the statistics of all candidate weak rules using current batch of
    /// training examples. 
    pub fn update(&mut self, data: &[ExampleInSampleSet]) -> Option<Tree> {
        // update global stats
        let weights = get_relative_weights(data);
        self.total_count += data.len();

        // preprocess examples
        let rho_gamma = self.rho_gamma.clone();
        let min_gamma = self.min_gamma;
        let data = {
            let mut data: Vec<(usize, f32, &Example, ([f32; 2], [f32; 2]), f32, f32)> =
                data.par_iter().zip(weights.par_iter()).map(|(example, weight)| {
                    let example = &example.0;
                    let index = self.tree.get_leaf_index(example);
                    let gamma = rho_gamma[index];
                    if gamma >= min_gamma {
                        let labeled_weight = weight * (example.label as f32);
                        let null_weight = 2.0 * gamma * weight;
                        let c_sq = ((1.0 + 2.0 * gamma) * weight).powi(2);
                        let left_score: Vec<_> =
                            LEFT_NODE.iter().map(|sign| sign * labeled_weight).collect();
                        let right_score: Vec<_> =
                            RIGHT_NODE.iter().map(|sign| sign * labeled_weight).collect();
                        let s_labeled_weight = (
                            [left_score[0], left_score[1]], [right_score[0], right_score[1]],
                        );
                        Some((index, *weight, example, s_labeled_weight, null_weight, c_sq))
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
        // Update gamma
        let shrink_threshold = self.num_examples_before_shrink;
        for k in 0..self.counts.len() {
            if self.counts[k] >= shrink_threshold {
                let old_rho_gamma = self.rho_gamma[k];
                let max_empirical_gamma = self.get_max_empirical_ratio(k) / 2.0;
                let new_rho_gamma = 0.9 * min(old_rho_gamma, max_empirical_gamma);
                self.reset(k, new_rho_gamma);
                debug!("shrink-gamma, {}, {}, {}, {}",
                       k, old_rho_gamma, max_empirical_gamma, self.rho_gamma[k]);
            }
        }

        // update each weak rule
        let range_start = self.range_start;
        let num_scanned = self.total_count;
        let num_candid = self.num_candid;
        let tree_node = self.bins.par_iter().zip(
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

                    let gamma = rho_gamma[index];
                    weak_rules_score[j][index][0] += labeled_weight[0];
                    weak_rules_score[j][index][1] += labeled_weight[1];

                    sum_c[j][index][0]            += labeled_weight[0] - null_weight;
                    sum_c[j][index][1]            += labeled_weight[1] - null_weight;

                    sum_c_squared[j][index][0]    += (1.0 + 2.0 * gamma).powi(2) * c_sq;
                    sum_c_squared[j][index][1]    += (1.0 + 2.0 * gamma).powi(2) * c_sq;
                });
            });

            // check stopping rule
            let mut valid_weak_rule = None;
            bin.get_vals().iter().enumerate().for_each(|(j, threshold)| {
                for index in 0..num_candid {
                    if is_zero(sum_c_squared[j][index][0]) {
                        // this candidate received no data or is already generated
                        continue;
                    }
                    for k in 0..2 {
                        let weak_rules_score = weak_rules_score[j][index][k];
                        let sum_c            = sum_c[j][index][k];
                        let sum_c_squared    = sum_c_squared[j][index][k];
                        let bound = get_bound(sum_c, sum_c_squared).unwrap_or(INFINITY);
                        if sum_c > bound {
                            let gamma = rho_gamma[index];
                            let base_pred = 0.5 * ((0.5 + gamma) / (0.5 - gamma)).ln();
                            valid_weak_rule = Some(
                                TreeNode {
                                    tree_index:     index,
                                    feature:        i + range_start,
                                    threshold:      *threshold,
                                    left_predict:   base_pred * LEFT_NODE[k],
                                    right_predict:  base_pred * RIGHT_NODE[k],

                                    gamma:          gamma,
                                    raw_martingale: weak_rules_score,
                                    sum_c:          sum_c,
                                    sum_c_squared:  sum_c_squared,
                                    bound:          bound,
                                    num_scanned:    num_scanned,
                                }
                            );
                        }
                    }
                }
            });
            valid_weak_rule
        }).find_any(|t| t.is_some()).unwrap_or(None);

        let mut ret = None;
        if tree_node.is_some() {
            let tree_node = tree_node.unwrap();
            tree_node.write_log();
            let (left_node, right_node) = self.tree.split(
                tree_node.tree_index, tree_node.feature, tree_node.threshold,
                tree_node.left_predict, tree_node.right_predict,
            );
            self.is_active[tree_node.tree_index] = false;
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
