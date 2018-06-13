use rayon::prelude::*;

use std::ops::Range;

use tree::Tree;
use commons::Example;
use commons::ExampleInSampleSet;
use super::bins::Bins;

use commons::max;
use commons::min;
use commons::get_bound;
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
type FloatTuple4 = (f32, f32, f32, f32);
type TupleTuple3 = (FloatTuple4, FloatTuple4, FloatTuple4);


// TODO: extend learner to support multi-level trees
/// A weak rule with an edge larger or equal to the targetting value of `gamma`
pub struct WeakRule {
    feature: usize,
    threshold: f32,
    left_predict: f32,
    right_predict: f32,

    raw_martingale: f32,
    sum_c: f32,
    sum_c_squared: f32,
    bound: f32
}

impl WeakRule {
    /// Return a decision tree (or decision stump) according to the valid weak rule
    pub fn create_tree(&self) -> Tree {
        debug!("new-tree-martingale, {}, {}, {}, {}",
               self.raw_martingale, self.sum_c, self.sum_c_squared, self.bound);
        let mut tree = Tree::new(2);
        tree.split(0, self.feature, self.threshold, self.left_predict, self.right_predict);
        tree.release();
        tree
    }
}


/// Statisitics of all weak rules that are being evaluated.
/// The objective of `Learner` is to find a weak rule that satisfies the condition of
/// the stopping rule.
pub struct Learner {
    bins: Vec<Bins>,
    range_start: usize,
    default_rho_gamma: f32,
    cur_rho_gamma: f32,

    weak_rules_score: ScoreBoard,
    sum_c:            ScoreBoard,
    sum_c_squared:    ScoreBoard,

    count: usize,
    sum_weights: f32,
    sum_weights_squared: f32,

    valid_weak_rule: Option<WeakRule>,
    outdated: bool
}

impl Learner {
    /// Create a `Learner` that search for valid weak rules.
    /// `default_gamma` is the initial value of the edge `gamma`.
    /// `bins` is vectors of the all thresholds on all candidate features for generating weak rules.
    ///
    /// `range` is the range of candidate features for generating weak rules. In most cases,
    /// if the algorithm is running on a single worker, `range` is 0..`num_of_features`;
    /// if the algorithm is running on multiple workers, `range` is a subset of the feature set.
    pub fn new(default_gamma: f32, bins: Vec<Bins>, range: &Range<usize>) -> Learner {
        let b1 = get_score_board(&bins);
        let b2 = get_score_board(&bins);
        let b3 = get_score_board(&bins);
        Learner {
            bins: bins,
            range_start: range.start,
            default_rho_gamma: default_gamma,
            cur_rho_gamma: default_gamma,

            weak_rules_score: b1,
            sum_c:            b2,
            sum_c_squared:    b3,

            count: 0,
            sum_weights: 0.0,
            sum_weights_squared: 0.0,

            valid_weak_rule: None,
            outdated: true
        }
    }

    /// Reset the statistics of all candidate weak rules,
    /// but leave the targetting `gamma` unchanged.
    pub fn reset(&mut self) {
        reset_score_board(&mut self.weak_rules_score);
        reset_score_board(&mut self.sum_c);
        reset_score_board(&mut self.sum_c_squared);

        self.count = 0;
        self.sum_weights = 0.0;
        self.sum_weights_squared = 0.0;
    }

    /// Return the number of examples scanned so far since detecting the last
    /// valid weak rule or since last reset.
    pub fn get_count(&self) -> usize {
        self.count
    }

    /// Return current targetting edge `gamma`
    pub fn get_gamma(&self) -> f32 {
        self.cur_rho_gamma
    }

    fn set_rho_gamma(&mut self, rho_gamma: f32) {
        self.cur_rho_gamma = rho_gamma;
        self.reset()
    }

    /// Reset the targetting `gamma` to `initial_gamma`, and reset all statistics
    /// of all candidate weak rules.
    pub fn reset_all(&mut self) {
        let rho_gamma = self.default_rho_gamma;
        self.set_rho_gamma(rho_gamma);
    }

    /// Shrinking the value of the targetting edge `gamma` of the valid weak rule
    pub fn shrink_target(&mut self) {
        let old_rho_gamma = self.cur_rho_gamma.clone();
        let max_empirical_ratio = self.get_max_empirical_ratio();
        self.cur_rho_gamma = min(self.cur_rho_gamma, max_empirical_ratio) * 0.9;
        self.reset();

        debug!("shrink-gamma, {}, {}", old_rho_gamma, self.cur_rho_gamma);
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
    pub fn update(&mut self, data: &[ExampleInSampleSet], weights: &Vec<f32>) {
        // update global stats
        let (sum_w, sum_w_squared) =
            weights.par_iter()
                   .cloned()
                   .map(|w| (w, w * w))
                   .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        self.sum_weights += sum_w;
        self.sum_weights_squared += sum_w_squared;
        self.count += data.len();

        // update each weak rule
        let gamma = self.cur_rho_gamma;
        self.update_weak_rules(
            data.par_iter().zip(weights.par_iter()).map(|(example, weight)| {
                // accumulate stats for each rule
                let label = get_symmetric_label(&example.0);
                let weighted_label = weight * label;
                let w_pos = (label - 2.0 * gamma) * weight;
                let w_neg = (-label - 2.0 * gamma) * weight;
                let w_sq = ((1.0 + 2.0 * gamma) * weight).powi(2);
                /*
                    1. Left +1, Right +1;
                    2. Left +1, Right -1;
                    3. Left -1, Right +1;
                    4. Left -1, Right -1.
                */
                let goes_to_left: TupleTuple3 = (
                    (weighted_label, weighted_label, -weighted_label, -weighted_label),
                    (w_pos, w_pos, w_neg, w_neg),
                    (w_sq, w_sq, w_sq, w_sq)
                );
                let goes_to_right: TupleTuple3 = (
                    (weighted_label, -weighted_label, weighted_label, -weighted_label),
                    (w_pos, w_neg, w_pos, w_neg),
                    (w_sq, w_sq, w_sq, w_sq)
                );
                (&example.0, goes_to_left, goes_to_right)
            }).collect()
        );

        self.outdated = true;
    }

    fn update_weak_rules(&mut self, examples: Vec<(&Example, TupleTuple3, TupleTuple3)>) {
        let range_start = self.range_start;
        self.bins.par_iter().zip(
            self.weak_rules_score.par_iter_mut().zip(
                self.sum_c.par_iter_mut().zip(
                    self.sum_c_squared.par_iter_mut()
                )
            )
        ).enumerate().for_each(
            |(i, (bin, (weak_rules_score, (sum_c, sum_c_squared))))| {
                examples.iter().for_each(|&(example, goes_to_left, goes_to_right)| {
                    let feature_val = example.get_features()[i + range_start] as f32;
                    bin.get_vals().iter().enumerate().for_each(|(j, threshold)| {
                        let direction =
                            if feature_val <= *threshold {
                                goes_to_left
                            } else {
                                goes_to_right
                            };
                        weak_rules_score[j][0] += (direction.0).0;
                        weak_rules_score[j][1] += (direction.0).1;
                        weak_rules_score[j][2] += (direction.0).2;
                        weak_rules_score[j][3] += (direction.0).3;

                        sum_c[j][0]            += (direction.1).0;
                        sum_c[j][1]            += (direction.1).1;
                        sum_c[j][2]            += (direction.1).2;
                        sum_c[j][3]            += (direction.1).3;

                        sum_c_squared[j][0]    += (direction.2).0;
                        sum_c_squared[j][1]    += (direction.2).1;
                        sum_c_squared[j][2]    += (direction.2).2;
                        sum_c_squared[j][3]    += (direction.2).3;
                    });
                });
        });
    }

    /// Return the valid weak rule if one is found, otherwise return `None`.
    pub fn get_new_weak_rule(&mut self) -> &Option<WeakRule> {
        if self.outdated {
            self.set_valid_weak_rule();
            self.outdated = false;
        }
        &self.valid_weak_rule
    }

    fn set_valid_weak_rule(&mut self) {
        let gamma = self.cur_rho_gamma;
        let ret =
            self.bins.par_iter().zip(
                self.weak_rules_score.par_iter().zip(
                    self.sum_c.par_iter().zip(
                        self.sum_c_squared.par_iter()
                    )
                )
            ).enumerate().map(
                |(i, (bin, (weak_rules_score, (sum_c, sum_c_squared))))| {
                    let mut ret = None;
                    let cur_bin = bin.get_vals();
                    for j in 0..cur_bin.len() {
                        let threshold = cur_bin[j];
                        for k in 0..4 {
                            let _sum_c = sum_c[j][k];
                            let _sum_c_squared = sum_c_squared[j][k];
                            let _score = weak_rules_score[j][k];
                            if let Some(bound) = get_bound(&_sum_c, &_sum_c_squared) {
                                if _sum_c > bound {
                                    let (left_predict, right_predict) = get_prediction(k, gamma);
                                    ret = Some(WeakRule {
                                        feature: i + self.range_start,
                                        threshold: threshold,
                                        left_predict: left_predict,
                                        right_predict: right_predict,

                                        raw_martingale: _score,
                                        sum_c: _sum_c,
                                        sum_c_squared: _sum_c_squared,
                                        bound: bound
                                    });
                                    break;
                                }
                            }
                        }
                        if ret.is_some() {
                            break;
                        }
                    }
                    ret
                }
            ).find_any(|r| r.is_some());
        match ret {
            Some(valid_weak_rule) => {
                self.valid_weak_rule = valid_weak_rule;
            },
            None                  => {
                self.valid_weak_rule = None;
            }
        }
    }
}


fn get_score_board(bins: &Vec<Bins>) -> ScoreBoard {
    let init_score_board = [0.0; 4];
    bins.iter().map(|bin| vec![init_score_board; bin.len()]).collect()
}

fn reset_score_board(score_board: &mut ScoreBoard) {
    score_board.par_iter_mut().for_each(|a| {
        a.iter_mut().for_each(|b| {
            b.iter_mut().for_each(|x| {
                *x = 0.0;
            });
        });
    });
}

#[inline]
fn get_prediction(rule_id: usize, rho_gamma: f32) -> (f32, f32) {
    /*
        1. Left +1, Right +1;
        2. Left +1, Right -1;
        3. Left -1, Right +1;
        4. Left -1, Right -1.
    */
    let pos_prediction = 0.5 * ((0.5 + rho_gamma) / (0.5 - rho_gamma)).ln();
    let neg_prediction = 0.5 * ((0.5 - rho_gamma) / (0.5 + rho_gamma)).ln();
    assert!(rule_id <= 3);
    match rule_id {
        0 => (pos_prediction, pos_prediction),
        1 => (pos_prediction, neg_prediction),
        2 => (neg_prediction, pos_prediction),
        _ => (neg_prediction, neg_prediction) // 3
    }
}
