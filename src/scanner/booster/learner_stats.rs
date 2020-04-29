use Example;
use super::learner::NUM_RULES;


/// Statisitics of all weak rules that are being evaluated.
/// The objective of `Learner` is to find a weak rule that satisfies the condition of
/// the stopping rule.
// [i][k][j][rule_id] => bin i slot j candidate-node k
// Stats from the feature level, i.e. split value index -> prediction/rule types
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
pub type CandidateNodeStats = Vec<Vec<EarlyStoppingStatsAtThreshold>>;

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
