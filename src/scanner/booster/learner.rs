use rayon::prelude::*;

use std::collections::HashMap;

use Example;
use commons::ExampleInSampleSet;
use commons::Model;
use commons::bins::Bins;

use commons::is_zero;
use super::learner_helpers;
use super::tree_node::TreeNode;

// TODO: The tree generation and score updates are for AdaBoost only,
// extend it to other potential functions

/*
TODO: extend support to regression tasks
TODO: re-use ScoreBoard space of the generated rules to reduce the memory footprint by half
      (just adding a mapping from the index here to the index in the tree should do the job)

ScoreBoard structure:
Split trree node -> Feature index -> split value index -> prediction/rule types

Each split corresponds to 2 types of predictions,
    1. Left +1, Right -1;
    2. Left -1, Right +1;
*/
pub const NUM_RULES: usize = 2;
pub const PREDS: [(f32, f32); NUM_RULES] = [(-1.0, 1.0), (1.0, -1.0)];

/// Statisitics of all weak rules that are being evaluated.
/// The objective of `Learner` is to find a weak rule that satisfies the condition of
/// the stopping rule.
// [i][k][j][rule_id] => bin i slot j candidate-node k
// Stats from the feature level, i.e. split value index -> prediction/rule types
#[derive(Serialize, Deserialize)]
pub struct EarlyStoppingStatsAtThreshold {
    // trackers for each candidate
    pub weak_rules_score: [f32; NUM_RULES],
    pub sum_c:            [f32; NUM_RULES],
    pub sum_c_squared:    [f32; NUM_RULES],
    // these trackers are same for all candidates, so they are redundant but this way we avoid
    // having to create another aggregator
    pub num_positive:     usize,
    pub num_negative:     usize,
    pub weight_positive:  f32,
    pub weight_negative:  f32,
    pub weight_squared:    f32,
}
impl EarlyStoppingStatsAtThreshold {
    pub fn new() -> EarlyStoppingStatsAtThreshold {
        EarlyStoppingStatsAtThreshold {
            weak_rules_score: [0.0; NUM_RULES],
            sum_c:            [0.0; NUM_RULES],
            sum_c_squared:    [0.0; NUM_RULES],

            num_positive:     0,
            num_negative:     0,
            weight_positive:  0.0,
            weight_negative:  0.0,
            weight_squared:   0.0,
        }
    }
}
// feature index -> split value index
type CandidateNodeStats = Vec<Vec<EarlyStoppingStatsAtThreshold>>;

#[derive(Serialize, Deserialize)]
pub struct EarlyStoppingIntermediate {
    pub left_score:      f32,
    pub right_score:     f32,
    // c = w * y * y_hat - 2 * \gamma * w
    pub left_c:          f32,
    pub right_c:         f32,
    // c_squared = c^2
    pub left_c_squared:  f32,
    pub right_c_squared: f32,
}
impl EarlyStoppingIntermediate {
    pub fn new(
        pred: &(f32, f32), example: &Example, weight: &f32, gamma: f32,
    ) -> EarlyStoppingIntermediate {
        let left_score  = pred.0 * weight * (example.label as f32);
        let right_score = pred.1 * weight * (example.label as f32);
        let null_hypothesis_score = 2.0 * gamma * weight;
        let left_c = left_score - null_hypothesis_score;
        let right_c = right_score - null_hypothesis_score;
        EarlyStoppingIntermediate {
            left_score:      left_score,
            right_score:     right_score,
            left_c:          left_c.clone(),
            right_c:         right_c.clone(),
            left_c_squared:  left_c * left_c,
            right_c_squared: right_c * right_c,
        }
    }

    pub fn zero() -> EarlyStoppingIntermediate {
        EarlyStoppingIntermediate {
            left_score:      0.0,
            right_score:     0.0,
            left_c:          0.0,
            right_c:         0.0,
            left_c_squared:  0.0,
            right_c_squared: 0.0,
        }
    }

    pub fn add(&mut self, other: &EarlyStoppingIntermediate) {
        self.left_score      += other.left_score;
        self.right_score     += other.right_score;
        self.left_c          += other.left_c;
        self.right_c         += other.right_c;
        self.left_c_squared  += other.left_c_squared;
        self.right_c_squared += other.right_c_squared;
    }

    pub fn add_right(&mut self, other: &EarlyStoppingIntermediate) {
        self.right_score     += other.right_score;
        self.right_c         += other.right_c;
        self.right_c_squared += other.right_c_squared;
    }

    pub fn move_to_left(&mut self, other: &EarlyStoppingIntermediate) {
        self.left_score      += other.left_score;
        self.left_c          += other.left_c;
        self.left_c_squared  += other.left_c_squared;
        self.right_score     -= other.right_score;
        self.right_c         -= other.right_c;
        self.right_c_squared -= other.right_c_squared;
    }
}
pub type RuleStats = Vec<EarlyStoppingIntermediate>;

#[derive(Serialize, Deserialize)]
pub struct Learner {
    bins: Vec<Bins>,
    num_features:   usize,
    _default_gamma: f32,
    min_gamma: f32,
    num_candid: usize,

    pub rho_gamma:    f32,
    pub expand_node:  usize,
    // global trackers
    pub total_count:  usize,
    total_weight:     f32,
    total_weight_sq:  f32,

    stats:            Vec<CandidateNodeStats>,
    // track the number of examples exposed to each splitting candidates
    // count:            usize,
    // sum_weight:       f32,
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
            num_candid: 1,

            rho_gamma:        default_gamma.clone(),
            expand_node:      0,
            total_count:      0,
            total_weight:     0.0,
            total_weight_sq:  0.0,

            stats:            vec![],
        };
        learner.reset();
        learner
    }

    /// Reset the statistics of all candidate weak rules
    /// (except gamma, because the advantage of the root node is not likely to increase)
    /// Trigger when the model or the gamma is changed
    pub fn reset(&mut self) {
        let get_candidate_node_stats = || {
            self.bins.iter().map(|bin| {
                (0..bin.len()).map(|_| {
                    EarlyStoppingStatsAtThreshold::new()
                }).collect()
            }).collect()
        };

        debug!("learner, learner is being reset, {}, {}, {}",
               self.expand_node, self.total_count, self.total_weight);
        self.stats = (0..self.num_candid).map(|_| get_candidate_node_stats())
                                         .collect();
        self.total_count = 0;
        self.total_weight = 0.0;
        self.total_weight_sq = 0.0;
    }

    pub fn get_max_empirical_ratio_tree_node(&self) -> Option<TreeNode> {
        type U = Vec<((usize, usize, usize, usize), f32)>;
        let total_weight = self.total_weight;
        let optimal_rule = {
            self.stats.iter().enumerate().flat_map(|(t, features)| {
                features.iter().enumerate().flat_map(|(i, thresholds)| {
                    thresholds.iter().enumerate().flat_map(|(j, stats)| {
                        stats.weak_rules_score.iter().enumerate()
                             .map(|(k, weak_rule_score)| {
                                 let rule_id = (t, i, j, k);
                                 let ratio = weak_rule_score / total_weight;
                                 (rule_id, ratio)
                             }).collect::<U>()
                    }).collect::<U>()
                }).collect::<U>()
            }).max_by(|(_k1, a), (_k2, b)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
        };
        if optimal_rule.is_none() {
            error!("Failed to find the best empirical rule.");
            return None
        }
        let (rule_id, ratio) = optimal_rule.unwrap();
        let (t, i, j, k) = rule_id;
        Some(learner_helpers::gen_tree_node(t, i, j, k, ratio))
    }

    pub fn is_gamma_significant(&self) -> bool {
        self.rho_gamma >= self.min_gamma
    }

    /// Update the statistics of all candidate weak rules using current batch of
    /// training examples.
    pub fn update(
        &mut self,
        tree: &Model,
        data: &[ExampleInSampleSet],
    ) -> Option<TreeNode> {
        // update global stats
        self.total_count       += data.len();
        self.total_weight      += data.par_iter().map(|t| (t.1).0).sum::<f32>();
        self.total_weight_sq   += data.par_iter().map(|t| ((t.1).0) * ((t.1).0)).sum::<f32>();
        self.num_candid         = tree.tree_size;

        let expand_node = self.expand_node;
        let rho_gamma = self.rho_gamma;

        // Put examples into bins by where they fall on the tree - Complexity: O(Examples)
        let mut data_by_node: HashMap<usize, Vec<(&Example, f32, RuleStats)>> = HashMap::new();
        // preprocess examples - Complexity: O(Examples * NumRules)
        let data: Vec<(usize, &Example, f32, RuleStats)> = learner_helpers::preprocess_data(
            data, tree, expand_node, rho_gamma);
        data.into_iter().for_each(|(index, example, weight, stats)| {
            data_by_node.entry(index).or_insert(Vec::new()).push((example, weight, stats));
        });

        // TODO: Calculate total sum of the weights and the number of examples

        // Update each weak rule - Complexity: O(Bins * Splits)
        let count = self.total_count;
        let total_weight = self.total_weight;
        let total_weight_sq = self.total_weight_sq;
        for index in 0..self.num_candid { // Splitting node candidate index
            if !data_by_node.contains_key(&index) {
                continue;
            }
            let data = &data_by_node[&index];
            let tree_node = {
                // all bins read data in parallel
                self.bins.par_iter()
                    .zip(
                        self.stats[index].par_iter_mut()
                    ).enumerate()
                    .map(|(feature_index, (feature_bins, mut stats))| {
                    learner_helpers::find_tree_node(
                        data, feature_index, rho_gamma, count, total_weight, total_weight_sq,
                        index, feature_bins, &mut stats)
                })
                .find_any(|t| t.is_some())
                .unwrap_or(None)
            };
            if tree_node.is_some() {
                return tree_node;
            }
        }
        None
    }

    pub fn set_gamma(&mut self, gamma: f32) -> bool {
        if !is_zero(gamma - self.rho_gamma) {
            debug!("set-gamma, {}, {}", self.rho_gamma, gamma);
            self.rho_gamma = gamma;
            self.reset();
            true
        } else {
            false
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
