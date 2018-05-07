use rayon::prelude::*;

use std::ops::Range;

use bins::Bins;
use tree::Tree;
use commons::Example;
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
pub struct WeakRule {
    feature: usize,
    threshold: f32,
    left_predict: f32,
    right_predict: f32,

    raw_martingale: f32,
    sum_c: f32,
    sum_c_squared: f32,
    bound: f32,

    left_positive: f32,
    left_negative: f32,
    right_positive: f32,
    right_negative: f32
}

impl WeakRule {
    pub fn create_tree(&self) -> Tree {
        debug!("new-tree-martingale, {}, {}, {}, {}, {}, {}, {}, {}",
               self.left_positive, self.left_negative, self.right_positive, self.right_negative,
               self.raw_martingale, self.sum_c, self.sum_c_squared, self.bound);
        let mut tree = Tree::new(2);
        tree.split(0, self.feature, self.threshold, self.left_predict, self.right_predict);
        tree.release();
        tree
    }
}

pub struct Learner {
    bins: Vec<Bins>,
    range_start: usize,
    default_rho_gamma: f32,
    cur_rho_gamma: f32,

    weak_rules_score: ScoreBoard,
    sum_c:            ScoreBoard,
    sum_c_squared:    ScoreBoard,

    left_positive:    ScoreBoard,
    left_negative:    ScoreBoard,
    right_positive:   ScoreBoard,
    right_negative:   ScoreBoard,

    count: usize,
    sum_weights: f32,
    sum_weights_squared: f32,

    valid_weak_rule: Option<WeakRule>,
    outdated: bool
}

impl Learner {
    pub fn new(default_rho_gamma: f32, bins: Vec<Bins>, range: &Range<usize>) -> Learner {
        let b1 = get_score_board(&bins);
        let b2 = get_score_board(&bins);
        let b3 = get_score_board(&bins);
        let b4 = get_score_board(&bins);
        let b5 = get_score_board(&bins);
        let b6 = get_score_board(&bins);
        let b7 = get_score_board(&bins);
        Learner {
            bins: bins,
            range_start: range.start,
            default_rho_gamma: default_rho_gamma,
            cur_rho_gamma: default_rho_gamma,

            weak_rules_score: b1,
            sum_c:            b2,
            sum_c_squared:    b3,

            left_positive:    b4,
            left_negative:    b5,
            right_positive:   b6,
            right_negative:   b7,

            count: 0,
            sum_weights: 0.0,
            sum_weights_squared: 0.0,

            valid_weak_rule: None,
            outdated: true
        }
    }

    pub fn reset(&mut self) {
        reset_score_board(&mut self.weak_rules_score);
        reset_score_board(&mut self.sum_c);
        reset_score_board(&mut self.sum_c_squared);

        reset_score_board(&mut self.left_positive);
        reset_score_board(&mut self.left_negative);
        reset_score_board(&mut self.right_positive);
        reset_score_board(&mut self.right_negative);

        self.count = 0;
        self.sum_weights = 0.0;
        self.sum_weights_squared = 0.0;
    }

    pub fn get_count(&self) -> usize {
        self.count
    }

    pub fn get_rho_gamma(&self) -> f32 {
        self.cur_rho_gamma
    }

    fn set_rho_gamma(&mut self, rho_gamma: f32) {
        self.cur_rho_gamma = rho_gamma;
        self.reset()
    }

    pub fn reset_all(&mut self) {
        let rho_gamma = self.default_rho_gamma;
        self.set_rho_gamma(rho_gamma);
    }

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

    pub fn update(&mut self, data: &Vec<Example>, weights: &Vec<f32>) {
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
                let label = get_symmetric_label(example);
                let weighted_label = weight * label;
                let w_pos = (label - 2.0 * gamma) * weight;
                let w_neg = (-label - 2.0 * gamma) * weight;
                // let w_sq = ((1.0 + 2.0 * gamma) * weight).powi(2);
                let w_sq = ((1.0 + 2.0 * gamma) * 1.0).powi(2);
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
                (example, goes_to_left, goes_to_right)
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
        ).zip(
            self.left_positive.par_iter_mut().zip(
                self.left_negative.par_iter_mut()
            ).zip(
                self.right_positive.par_iter_mut().zip(
                    self.right_negative.par_iter_mut()
                )
            )
        ).enumerate().for_each(
            |(i, ((bin, (weak_rules_score, (sum_c, sum_c_squared))),
                  ((left_pos, left_neg), (right_pos, right_neg))))| {
                examples.iter().for_each(|&(example, goes_to_left, goes_to_right)| {
                    let feature_val = example.get_features()[i + range_start] as f32;
                    let label = get_symmetric_label(example);
                    let weight = (goes_to_left.0).0 * label;
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

                        if feature_val <= *threshold {
                            if label > 0.0 {
                                left_pos[j][0] += weight;
                                left_pos[j][1] += weight;
                                left_pos[j][2] += weight;
                                left_pos[j][3] += weight;
                            } else {
                                left_neg[j][0] += weight;
                                left_neg[j][1] += weight;
                                left_neg[j][2] += weight;
                                left_neg[j][3] += weight;
                            }
                        } else {
                            if label > 0.0 {
                                right_pos[j][0] += weight;
                                right_pos[j][1] += weight;
                                right_pos[j][2] += weight;
                                right_pos[j][3] += weight;
                            } else {
                                right_neg[j][0] += weight;
                                right_neg[j][1] += weight;
                                right_neg[j][2] += weight;
                                right_neg[j][3] += weight;
                            }
                        }
                    });
                });
        });
    }

    pub fn get_new_weak_rule(&mut self, cur_best: bool) -> &Option<WeakRule> {
        if self.outdated || (cur_best && self.valid_weak_rule.is_none()) {
            self.set_valid_weak_rule(cur_best);
            self.outdated = false;
        }
        &self.valid_weak_rule
    }

    fn set_valid_weak_rule(&mut self, cur_best: bool) {
        let mut max_score = 0.0;
        if cur_best {
            for i in 0..self.weak_rules_score.len() {
                for j in 0..self.weak_rules_score[i].len() {
                    for k in 0..4 {
                        if self.weak_rules_score[i][j][k] > max_score {
                            max_score = self.weak_rules_score[i][j][k];
                        }
                    }
                }
            }
        }
        let gamma = self.cur_rho_gamma;
        let sum_weights = self.sum_weights;
        let ret =
            self.bins.par_iter().zip(
                self.weak_rules_score.par_iter().zip(
                    self.sum_c.par_iter().zip(
                        self.sum_c_squared.par_iter()
                    )
                )
            ).zip(
                self.left_positive.par_iter().zip(
                    self.left_negative.par_iter()
                ).zip(
                    self.right_positive.par_iter().zip(
                        self.right_negative.par_iter()
                    )
                )
            ).enumerate().map(
                |(i, ((bin, (weak_rules_score, (sum_c, sum_c_squared))),
                      ((left_positive, left_negative), (right_positive, right_negative))))| {
                    let mut ret = None;
                    let cur_bin = bin.get_vals();
                    for j in 0..cur_bin.len() {
                        let threshold = cur_bin[j];
                        for k in 0..4 {
                            let _sum_c = sum_c[j][k];
                            let _sum_c_squared = sum_c_squared[j][k];
                            let _score = weak_rules_score[j][k];

                            let _left_pos = left_positive[j][k];
                            let _left_neg = left_negative[j][k];
                            let _right_pos = right_positive[j][k];
                            let _right_neg = right_negative[j][k];

                            let some_bound = get_bound(&_sum_c, &_sum_c_squared);
                            if cur_best && _score >= max_score || some_bound.is_some() {
                                let bound = if !cur_best {
                                    some_bound.unwrap()
                                } else {
                                    -1.0
                                };
                                if _sum_c > bound {
                                    let _gamma = if bound < 0.0 {
                                        _score / sum_weights / 2.0
                                    } else {
                                        gamma
                                    };
                                    let (left_predict, right_predict) = get_prediction(k, _gamma);
                                    ret = Some(WeakRule {
                                        feature: i + self.range_start,
                                        threshold: threshold,
                                        left_predict: left_predict,
                                        right_predict: right_predict,

                                        raw_martingale: _score,
                                        sum_c: _sum_c,
                                        sum_c_squared: _sum_c_squared,
                                        bound: bound,

                                        left_positive: _left_pos,
                                        left_negative: _left_neg,
                                        right_positive: _right_pos,
                                        right_negative: _right_neg
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
