use rayon::prelude::*;

use std::f32::INFINITY;
use std::ops::Range;

use commons::ExampleInSampleSet;
use tree::Tree;
use super::bins::Bins;
use super::super::Example;

use buffer_loader::BufferLoader;
use commons::max;
use commons::min;
use commons::get_bound;
use commons::get_relative_weights;
use commons::get_symmetric_label;

/*
TODO: extend support to regression tasks

Each split corresponds to 4 types of predictions,
    1. Left +1, Right +1;
    2. Left +1, Right -1;
    3. Left -1, Right +1;
    4. Left -1, Right -1.
*/
type ScoreBoard = Vec<Vec<[f32; 4]>>;

const LEFT_NODE:  [f32; 4] = [1.0, 1.0, -1.0, -1.0];
const RIGHT_NODE: [f32; 4] = [1.0, -1.0, 1.0, -1.0];


// TODO: extend learner to support multi-level trees
/// A weak rule with an edge larger or equal to the targetting value of `gamma`
pub struct WeakRule {
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

impl WeakRule {
    /// Return a decision tree (or decision stump) according to the valid weak rule
    pub fn to_tree(self) -> Tree {
        let mut tree = Tree::new(2);
        tree.split(0, self.feature, self.threshold, self.left_predict, self.right_predict);
        tree.release();
        tree
    }

    pub fn write_log(&self, model_len: usize, curr_sum_gamma: f32) {
        info!(
            "new-tree-info, {}, {}, {}, {}, {}, {}, {}, {}",
            model_len + 1,
            self.num_scanned,
            self.gamma,
            self.raw_martingale,
            self.sum_c,
            self.sum_c_squared,
            self.bound,
            curr_sum_gamma + self.gamma,
        );
    }
}


/// Statisitics of all weak rules that are being evaluated.
/// The objective of `Learner` is to find a weak rule that satisfies the condition of
/// the stopping rule.
pub struct Learner {
    bins: Vec<Bins>,
    range_start: usize,
    cur_rho_gamma: f32,
    num_examples_before_shrink: usize,

    weak_rules_score: ScoreBoard,
    sum_c:            ScoreBoard,
    sum_c_squared:    ScoreBoard,

    pub count: usize,
    sum_weights: f32,
    sum_weights_squared: f32,

}

impl Learner {
    /// Create a `Learner` that search for valid weak rules.
    /// `default_gamma` is the initial value of the edge `gamma`.
    /// `bins` is vectors of the all thresholds on all candidate features for generating weak rules.
    ///
    /// `range` is the range of candidate features for generating weak rules. In most cases,
    /// if the algorithm is running on a single worker, `range` is 0..`num_of_features`;
    /// if the algorithm is running on multiple workers, `range` is a subset of the feature set.
    pub fn new(default_gamma: f32, num_examples_before_shrink: u32, bins: Vec<Bins>, range: &Range<usize>) -> Learner {
        let b = [0.0; 4];
        Learner {
            range_start: range.start,
            cur_rho_gamma: default_gamma,
            num_examples_before_shrink: num_examples_before_shrink as usize,

            weak_rules_score: bins.iter().map(|bin| vec![b; bin.len()]).collect(),
            sum_c:            bins.iter().map(|bin| vec![b; bin.len()]).collect(),
            sum_c_squared:    bins.iter().map(|bin| vec![b; bin.len()]).collect(),

            count: 0,
            sum_weights: 0.0,
            sum_weights_squared: 0.0,
            bins: bins,
        }
    }

    /// Reset the statistics of all candidate weak rules,
    /// but leave the targetting `gamma` unchanged.
    pub fn reset(&mut self) {
        for i in 0..self.weak_rules_score.len() {
            for j in 0..self.weak_rules_score[i].len() {
                for k in 0..4 {
                    self.weak_rules_score[i][j][k] = 0.0;
                    self.sum_c[i][j][k]            = 0.0;
                    self.sum_c_squared[i][j][k]    = 0.0;
                }
            }
        }

        self.count = 0;
        self.sum_weights = 0.0;
        self.sum_weights_squared = 0.0;
    }

    fn get_max_empirical_ratio(&self) -> f32 {
        self.weak_rules_score.iter().flat_map(|rules| {
            rules.iter().flat_map(|scores| {
                scores.iter().map(|s| s / self.sum_weights)
            })
        }).fold(0.0, max)
    }

    /// Update the statistics of all candidate weak rules using current batch of
    /// training examples. 
    pub fn update(&mut self, data: &[ExampleInSampleSet]) -> Option<WeakRule> {
        // Shrinking the value of the targetting edge `gamma` if it was too high
        if self.count >= self.num_examples_before_shrink {
            let old_rho_gamma = self.cur_rho_gamma;
            let max_empirical_gamma = self.get_max_empirical_ratio() / 2.0;
            self.cur_rho_gamma = 0.9 * min(self.cur_rho_gamma, max_empirical_gamma);
            self.reset();
            debug!("shrink-gamma, {}, {}, {}",
                   old_rho_gamma, max_empirical_gamma, self.cur_rho_gamma);
        }

        // update global stats
        let weights = get_relative_weights(data);
        let (sum_w, sum_w_squared) =
            weights.par_iter()
                   .map(|x| (x.clone(), (x * x).clone()))
                   .reduce(|| (0.0, 0.0), |(a1, a2), (b1, b2)| (a1 + b1, a2 + b2));
        self.sum_weights += sum_w;
        self.sum_weights_squared += sum_w_squared;
        self.count += data.len();

        // preprocess examples
        let gamma = self.cur_rho_gamma;
        let data: Vec<(&Example, ([f32; 4], [f32; 4]))> =
            data.par_iter().zip(weights.par_iter()).map(|(example, weight)| {
                let labeled_weight = weight * get_symmetric_label(&(example.0));
                // let null_weight = 2.0 * gamma * weight;
                // let c_sq = ((1.0 + 2.0 * gamma) * weight).powi(2);
                let left_score: Vec<_> =
                    LEFT_NODE.iter().map(|sign| sign * labeled_weight).collect();
                let right_score: Vec<_> =
                    RIGHT_NODE.iter().map(|sign| sign * labeled_weight).collect();
                let signed_labeled_weight = (
                    [left_score[0], left_score[1], left_score[2], left_score[3]],
                    [right_score[0], right_score[1], right_score[2], right_score[3]],
                );
                (&example.0, signed_labeled_weight)
            }).collect();

        // update each weak rule
        let range_start = self.range_start;
        let num_scanned = self.count;
        self.bins.par_iter().zip(
            self.weak_rules_score.par_iter_mut()
        ).zip(
            self.sum_c.par_iter_mut()
        ).zip(
            self.sum_c_squared.par_iter_mut()
        ).enumerate().map(|(i, (((bin, weak_rules_score), sum_c), sum_c_squared))| {
            // Update stats
            sum_c.iter_mut().for_each(|v| {
                v[0] -= 2.0 * gamma * sum_w;
                v[1] -= 2.0 * gamma * sum_w;
                v[2] -= 2.0 * gamma * sum_w;
                v[3] -= 2.0 * gamma * sum_w;
            });
            sum_c_squared.iter_mut().for_each(|v| {
                v[0] += (1.0 + 2.0 * gamma).powi(2) * sum_w_squared;
                v[1] += (1.0 + 2.0 * gamma).powi(2) * sum_w_squared;
                v[2] += (1.0 + 2.0 * gamma).powi(2) * sum_w_squared;
                v[3] += (1.0 + 2.0 * gamma).powi(2) * sum_w_squared;
            });
            data.iter().for_each(|(example, labeled_weight)| {
                let feature_val = example.feature[i + range_start] as f32;
                bin.get_vals().iter().enumerate().for_each(|(j, threshold)| {
                    let labeled_weight =
                        if feature_val <= *threshold {
                            &labeled_weight.0
                        } else {
                            &labeled_weight.1
                        };

                    weak_rules_score[j][0] += labeled_weight[0];
                    weak_rules_score[j][1] += labeled_weight[1];
                    weak_rules_score[j][2] += labeled_weight[2];
                    weak_rules_score[j][3] += labeled_weight[3];

                    sum_c[j][0]            += labeled_weight[0];  // - null_weight;
                    sum_c[j][1]            += labeled_weight[1];  // - null_weight;
                    sum_c[j][2]            += labeled_weight[2];  // - null_weight;
                    sum_c[j][3]            += labeled_weight[3];  // - null_weight;
                });
            });

            // check stopping rule
            let mut valid_weak_rule = None;
            bin.get_vals().iter().enumerate().for_each(|(j, threshold)| {
                for k in 0..4 {
                    let weak_rules_score = weak_rules_score[j][k];
                    let sum_c            = sum_c[j][k];
                    let sum_c_squared    = sum_c_squared[j][k];
                    let bound = get_bound(sum_c, sum_c_squared).unwrap_or(INFINITY);
                    if sum_c > bound {
                        let base_pred = 0.5 * ((0.5 + gamma) / (0.5 - gamma)).ln();
                        valid_weak_rule = Some(
                            WeakRule {
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
            });
            valid_weak_rule
        }).find_any(|t| t.is_some()).unwrap_or(None)
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
