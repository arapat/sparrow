
use TFeature;


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