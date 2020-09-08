use rayon::prelude::*;

use std::collections::HashMap;

use Example;
use TFeature;
use commons::ExampleInSampleSet;
use commons::tree::Tree;
use commons::bins::Bins;

use commons::is_zero;
use super::learner_helpers;

// TODO: The tree generation and score updates are for AdaBoost only,
// extend it to other potential functions

/*
TODO: extend support to regression tasks
TODO: re-use ScoreBoard space of the generated rules to reduce the memory footprint by half
      (just adding a mapping from the index here to the index in the tree should do the job)

ScoreBoard structure:
(candidate splitting node index) -> (3-level Scoreboard)
(3-level ScoreBoard): Feature index -> split value index -> prediction types

Each split corresponds to 2 types of predictions,
    1. Left +1, Right -1;
    2. Left -1, Right +1;
*/
pub const NUM_PREDS: usize = 2;
const NUM_RULES: usize = NUM_PREDS;
pub const PREDS: [(f32, f32); NUM_PREDS] = [(-1.0, 1.0), (1.0, -1.0)];
// (pred1, false), (pred1, true), (pred2, false), (pred2, true)
type ScoreBoard = Vec<Vec<[f32; NUM_RULES]>>;
type ScoreBoard1 = Vec<Vec<f32>>;
// (f32, f32) -> Stats if falls under left, and if falls under right
pub type RuleStats = [[(f32, f32); 2]; NUM_PREDS];


/// A weak rule with an edge larger or equal to the targetting value of `gamma`
pub struct TreeNode {
    pub prt_index: usize,
    pub feature: usize,
    pub threshold: TFeature,
    pub predict: (f32, f32),

    pub gamma: f32,
    pub raw_martingale: f32,
    pub sum_c: f32,
    pub sum_c_squared: f32,
    pub bound: f32,
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
    pub num_features:   usize,
    num_candid: usize,

    pub rho_gamma:        f32,
    pub _expand_node:      usize,
    // global trackers
    pub total_count:  usize,
    total_weight:     f32,
    total_weight_sq:  f32,
    // trackers for each candidate
    weak_rules_score: Vec<ScoreBoard>,
    sum_c_squared:    Vec<ScoreBoard>,
    // trackers for debugging
    num_positive:     Vec<ScoreBoard1>,
    num_negative:     Vec<ScoreBoard1>,
    weight_positive:  Vec<ScoreBoard1>,
    weight_negative:  Vec<ScoreBoard1>,
    // track the number of examples exposed to each splitting candidates
    // count:            usize,
    // sum_weight:       f32,
}

impl Learner {
    /// Create a `Learner` that search for valid weak rules.
    /// `gamma` is the initial value of the edge `gamma`.
    /// `bins` is vectors of the all thresholds on all candidate features for generating weak rules.
    pub fn new(
        gamma: f32,
        bins: Vec<Bins>,
        num_features: usize,
        num_splits: usize,
    ) -> Learner {
        let mut learner = Learner {
            bins: bins,
            num_features: num_features.clone(),
            num_candid: 0,

            rho_gamma:        gamma.clone(),
            _expand_node:      0,
            total_count:      0,
            total_weight:     0.0,
            total_weight_sq:  0.0,
            weak_rules_score: vec![],
            sum_c_squared:    vec![],

            num_positive:     vec![],
            num_negative:     vec![],
            weight_positive:  vec![],
            weight_negative:  vec![],
        };
        let bin_size: Vec<usize> = learner.bins.iter().map(|bin| bin.len()).collect();
        let new_scoreboard =
            || bin_size.iter().map(|size| vec![[0.0; NUM_RULES]; *size]).collect();
        let new_scoreboard1 = || bin_size.iter().map(|size| vec![0.0; *size]).collect();
        let num_nodes = (num_splits + 1) * 2 - 1;
        for _ in 0..num_nodes {
            learner.weak_rules_score.push(new_scoreboard());
            learner.sum_c_squared.push(new_scoreboard());
            learner.num_positive.push(new_scoreboard1());
            learner.num_negative.push(new_scoreboard1());
            learner.weight_positive.push(new_scoreboard1());
            learner.weight_negative.push(new_scoreboard1());
        }
        learner
    }

    /// Reset the statistics of all candidate weak rules
    /// (except gamma, because the advantage of the root node is not likely to increase)
    /// Trigger when the model or the gamma is changed
    pub fn reset(&mut self) {
        for t in 0..self.num_candid {
            for i in 0..self.num_features {
                for j in 0..self.bins[i].len() {
                    for k in 0..NUM_RULES {
                        self.weak_rules_score[t][i][j][k] = 0.0;
                        self.sum_c_squared[t][i][j][k]    = 0.0;
                    }
                    self.num_positive[t][i][j] = 0.0;
                    self.num_negative[t][i][j] = 0.0;
                    self.weight_positive[t][i][j] = 0.0;
                    self.weight_negative[t][i][j] = 0.0;
                }
            }
        }
        trace!("learner, learner is reset, {}, {}, {}",
               self._expand_node, self.total_count, self.total_weight);
        self.total_count = 0;
        self.total_weight = 0.0;
        self.total_weight_sq = 0.0;
    }

    pub fn get_max_empirical_ratio_tree_node(&self) -> TreeNode {
        let mut max_ratio = 0.0;
        let mut actual_ratio = 0.0;
        let mut rule_id = (0, 0, 0, 0);
        for t in 0..self.num_candid {
            for i in 0..self.num_features {
                for j in 0..self.bins[i].len() {
                    for k in 0..NUM_RULES {
                        // max ratio considers absent examples, actual ratio does not
                        let ratio = self.weak_rules_score[t][i][j][k] / self.total_weight;
                        if ratio >= max_ratio {
                            max_ratio = ratio;
                            actual_ratio = ratio;
                            rule_id = (t, i, j, k);
                        }
                    }
                }
            }
        }

        let (t, i, j, k) = rule_id;
        let mut tree_node = learner_helpers::gen_tree_node(t, i, j, k, actual_ratio);
        let weak_rules_score = self.weak_rules_score[t][i][j][k];
        let gamma = self.rho_gamma;
        tree_node.raw_martingale  = weak_rules_score;
        tree_node.sum_c           = weak_rules_score - 2.0 * gamma * self.total_weight;
        tree_node.sum_c_squared   = self.sum_c_squared[t][i][j][k]
                                    + 4.0 * gamma * gamma * self.total_weight_sq;
        tree_node.bound           = self.total_weight;
        tree_node.num_scanned     = self.total_count;
        tree_node.positive        = self.num_positive[t][i][j] as usize;
        tree_node.negative        = self.num_negative[t][i][j] as usize;
        tree_node.positive_weight = self.weight_positive[t][i][j];
        tree_node.negative_weight = self.weight_negative[t][i][j];
        tree_node
    }

    /// Update the statistics of all candidate weak rules using current batch of
    /// training examples.
    pub fn update(
        &mut self,
        tree: &Tree,
        data: &[ExampleInSampleSet],
    ) -> Option<TreeNode> {
        // update global stats
        self.total_count       += data.len();
        self.total_weight      += data.par_iter().map(|t| (t.1).0).sum::<f32>();
        self.total_weight_sq   += data.par_iter().map(|t| ((t.1).0) * ((t.1).0)).sum::<f32>();
        self.num_candid         = tree.num_nodes;

        let rho_gamma = self.rho_gamma;

        // preprocess examples - Complexity: O(Examples * NumRules)
        let data: Vec<(usize, f32, (&Example, RuleStats))> = learner_helpers::preprocess_data(
            data, tree, rho_gamma);

        // Put examples into bins by where they fall on the tree - Complexity: O(Examples)
        let mut data_by_node: HashMap<usize, Vec<(f32, (&Example, RuleStats))>> = HashMap::new();
        data.into_iter().for_each(|(index, weight, stats)| {
            data_by_node.entry(index).or_insert(Vec::new()).push((weight, stats));
        });

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
                self.bins.par_iter().zip(
                    self.weak_rules_score[index].par_iter_mut()
                ).zip(
                    self.sum_c_squared[index].par_iter_mut()
                ).zip(
                    self.num_positive[index].par_iter_mut()
                        .zip(self.num_negative[index].par_iter_mut())
                        .zip(self.weight_positive[index].par_iter_mut())
                        .zip(self.weight_negative[index].par_iter_mut())
                ).enumerate()
                .map(|(i, zipped_values)| {
                    let (((bin, weak_rules_score), sum_c_squared), debug_info) = zipped_values;
                    learner_helpers::find_tree_node(
                        &data, i, rho_gamma, count, total_weight, total_weight_sq,
                        index, bin, weak_rules_score, sum_c_squared, debug_info)
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
        if expand_node != self._expand_node {
            debug!("set-expand-node, {}, {}", self._expand_node, expand_node);
            self._expand_node = expand_node;
            true
        } else {
            false
        }
    }
}
