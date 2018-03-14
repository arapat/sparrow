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

    martingale: f32,
    bound: f32
}

impl WeakRule {
    pub fn create_tree(&self) -> Tree {
        let mut tree = Tree::new(2);
        tree.split(0, self.feature, self.threshold, self.left_predict, self.right_predict);
        tree.release();
        tree
    }
}

pub struct Learner {
    feature_size: usize,
    bins: Vec<Bins>,
    num_weak_rules: usize,

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
    pub fn new(feature_size: usize, default_rho_gamma: f32, bins: Vec<Bins>) -> Learner {
        let num_weak_rules = bins.iter().map(|bin| bin.len()).sum();
        let b1 = get_score_board(&bins);
        let b2 = get_score_board(&bins);
        let b3 = get_score_board(&bins);
        Learner {
            feature_size: feature_size,
            bins: bins,
            num_weak_rules: num_weak_rules,

            default_rho_gamma: default_rho_gamma,
            cur_rho_gamma: default_rho_gamma,

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

    pub fn reset(&mut self) {
        reset_score_board(&mut self.weak_rules_score);
        reset_score_board(&mut self.sum_c);
        reset_score_board(&mut self.sum_c_squared);

        self.count = 0;
        self.sum_weights = 0.0;
        self.sum_weights_squared = 0.0;
    }

    pub fn get_count(&self) -> usize {
        self.count
    }

    pub fn get_ess(&self) -> f32 {
        if self.count <= 0 {
            1.0
        } else {
            self.sum_weights.powi(2) / self.sum_weights_squared / (self.count as f32)
        }
    }

    pub fn get_rho_gamma(&self) -> f32 {
        self.cur_rho_gamma
    }

    pub fn set_rho_gamma(&mut self, rho_gamma: f32) {
        self.cur_rho_gamma = rho_gamma;
        self.reset()
    }

    pub fn shrink_target(&mut self) {
        let old_rho_gamma = self.cur_rho_gamma.clone();
        let max_empirical_ratio = self.get_max_empirical_ratio();
        self.cur_rho_gamma = min(self.cur_rho_gamma, max_empirical_ratio) * 0.9;
        self.reset();

        info!("Target advantage is shrinked: {} -> {}.", old_rho_gamma, self.cur_rho_gamma);
    }

    fn get_max_empirical_ratio(&self) -> f32 {
        self.weak_rules_score.iter().flat_map(|rules| {
            rules.iter().flat_map(|scores| {
                scores.iter().map(|s| s / self.sum_weights)
            })
        }).fold(0.0, max)
    }

    pub fn update(&mut self, data: &Vec<Example>, weights: &Vec<f32>) {
        // update stats
        self.count += data.len();
        data.iter().zip(weights.iter()).for_each(|(example, weight)| {
            // accumulate global stats
            self.sum_weights += weight;
            self.sum_weights_squared += weight * weight;

            // accumulate stats for each rule
            let label = get_symmetric_label(example);
            let feature = example.get_features();
            let weighted_label = weight * label;
            let w_pos = (label - 2.0 * self.cur_rho_gamma) * weight;
            let w_neg = (-label - 2.0 * self.cur_rho_gamma) * weight;
            let w_sq = (1.0 + 2.0 * self.cur_rho_gamma).powi(2);
            // let wp_sq = w_pos.powi(2);
            // let wn_sq = w_neg.powi(2);
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
            (0..self.bins.len()).for_each(|i| {
                (0..self.bins[i].get_vals().len()).for_each(|j| {
                    let direction =
                        if feature[i] <= self.bins[i].get_vals()[j] {
                            goes_to_left
                        } else {
                            goes_to_right
                        };
                    self.weak_rules_score[i][j][0] += (direction.0).0;
                    self.weak_rules_score[i][j][1] += (direction.0).1;
                    self.weak_rules_score[i][j][2] += (direction.0).2;
                    self.weak_rules_score[i][j][3] += (direction.0).3;

                    self.sum_c[i][j][0]            += (direction.1).0;
                    self.sum_c[i][j][1]            += (direction.1).1;
                    self.sum_c[i][j][2]            += (direction.1).2;
                    self.sum_c[i][j][3]            += (direction.1).3;

                    self.sum_c_squared[i][j][0]    += (direction.2).0;
                    self.sum_c_squared[i][j][1]    += (direction.2).1;
                    self.sum_c_squared[i][j][2]    += (direction.2).2;
                    self.sum_c_squared[i][j][3]    += (direction.2).3;
                });
            });
        });

        self.outdated = true;
    }

    pub fn get_new_weak_rule(&mut self) -> &Option<WeakRule> {
        if self.outdated {
            self.set_valid_weak_rule();
            self.outdated = false;
        }
        &self.valid_weak_rule
    }

    fn set_valid_weak_rule(&mut self) {
        self.valid_weak_rule = None;
        (0..self.bins.len()).for_each(|i| {
            (0..self.bins[i].get_vals().len()).for_each(|j| {
                (0..4).for_each(|k| {
                    let sum_c = &self.sum_c[i][j][k];
                    let sum_c_squared = &self.sum_c_squared[i][j][k];
                    match get_bound(sum_c, sum_c_squared) {
                        Some(bound) => {
                            if *sum_c > bound {
                                let (left_predict, right_predict) = get_prediction(k, self.cur_rho_gamma);
                                let threshold = self.bins[i].get_vals()[j];
                                // debug!("Weak rule is detected. sum_c={:?}, sum_c_sq={:?}, \
                                //         bound={}, advantage={}. feature={}, threshold={}, type={}.",
                                //        self.sum_c[i][j], self.sum_c_squared[i][j], bound,
                                //        self.cur_rho_gamma, i, threshold, k);

                                self.valid_weak_rule = Some(WeakRule {
                                    feature: i,
                                    threshold: threshold,
                                    left_predict: left_predict,
                                    right_predict: right_predict,

                                    martingale: *sum_c,
                                    bound: bound
                                });
                            }
                        },
                        None => {}
                    }
                });
            });
        });
    }
}


fn get_score_board(bins: &Vec<Bins>) -> ScoreBoard {
    let init_score_board = [0.0; 4];
    bins.iter().map(|bin| vec![init_score_board; bin.len()]).collect()
}

fn reset_score_board(score_board: &mut ScoreBoard) {
    score_board.iter_mut().for_each(|a| {
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
