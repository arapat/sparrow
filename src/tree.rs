use rayon::prelude::*;

use commons::Example;

type DimScaleType = u16;


/*
Why JSON (rather than binary)?
    - Readable for human
    - Compatible with Python
    - BufReader-friendly by using newline as separator
*/
#[derive(Serialize, Deserialize, Debug)]
pub struct Tree {
    max_leaves:     DimScaleType,
    num_leaves:     DimScaleType,
    left_child:     Vec<DimScaleType>,
    right_child:    Vec<DimScaleType>,
    split_feature:  Vec<Option<DimScaleType>>,
    threshold:      Vec<f32>,
    leaf_value:     Vec<f32>,
    leaf_depth:     Vec<DimScaleType>
    // leaf_parent:    Vec<DimScaleType>,
    // leaf_count:     Vec<DimScaleType>,
    // internal_value: Vec<DimScaleType>,
    // internal_count: Vec<DimScaleType>,
}

impl Clone for Tree {
    fn clone(&self) -> Tree {
        Tree {
            max_leaves:     self.max_leaves,
            num_leaves:     self.num_leaves,
            left_child:     self.left_child.clone(),
            right_child:    self.right_child.clone(),
            split_feature:  self.split_feature.clone(),
            threshold:      self.threshold.clone(),
            leaf_value:     self.leaf_value.clone(),
            leaf_depth:     self.leaf_depth.clone()
        }
    }
}

impl Tree {
    pub fn new(max_leaves: DimScaleType) -> Tree {
        let max_nodes = max_leaves * 2;
        let mut tree = Tree {
            max_leaves:     max_leaves,
            num_leaves:     0,
            left_child:     Vec::with_capacity(max_nodes as usize),
            right_child:    Vec::with_capacity(max_nodes as usize),
            split_feature:  Vec::with_capacity(max_nodes as usize),
            threshold:      Vec::with_capacity(max_nodes as usize),
            leaf_value:     Vec::with_capacity(max_nodes as usize),
            leaf_depth:     Vec::with_capacity(max_nodes as usize)
            // leaf_parent:    Vec::with_capacity(max_leaves),
            // leaf_count:     Vec::with_capacity(max_leaves),
            // internal_value: Vec::with_capacity(max_leaves as usize),
            // internal_count: Vec::with_capacity(max_leaves),
        };
        tree.add_new_node(0.0, 0);
        tree
    }

    pub fn release(&mut self) {
        self.left_child.shrink_to_fit();
        self.right_child.shrink_to_fit();
        self.split_feature.shrink_to_fit();
        self.threshold.shrink_to_fit();
        self.leaf_value.shrink_to_fit();
        self.leaf_depth.shrink_to_fit();
    }

    pub fn split(&mut self, leaf: usize, feature: usize, threshold: f32,
                 left_value: f32, right_value: f32) {
        let leaf_value = self.leaf_value[leaf];
        let leaf_depth = self.leaf_depth[leaf];

        self.split_feature[leaf] = Some(feature as DimScaleType);
        self.threshold[leaf] = threshold;
        self.left_child[leaf] = self.num_leaves;
        self.add_new_node(leaf_value + left_value, leaf_depth + 1);
        self.right_child[leaf] = self.num_leaves;
        self.add_new_node(leaf_value + right_value, leaf_depth + 1);
    }

    pub fn add_prediction_to_score(
            &self, data: &Vec<Example>, score: &mut [f32]) {
        score.par_iter_mut()
             .zip(
                 data.par_iter()
                     .map(|ex| self.get_leaf_prediction(ex)))
             .for_each(|(accum, update)| *accum += update)
    }

    fn get_leaf_prediction(&self, data: &Example) -> f32 {
        let mut node: usize = 0;
        let features = data.get_features();
        while let Some(split_feature) = self.split_feature[node] {
            node = if features[split_feature as usize] as f32 <= self.threshold[node] {
                self.left_child[node]
            } else {
                self.right_child[node]
            } as usize;
        }
        self.leaf_value[node]
    }

    fn add_new_node(&mut self, leaf_value: f32, depth: DimScaleType) {
        self.num_leaves += 1;
        self.left_child.push(0);
        self.right_child.push(0);
        self.split_feature.push(None);
        self.threshold.push(0.0);
        self.leaf_value.push(leaf_value);
        self.leaf_depth.push(depth);
    }

    #[allow(dead_code)]
    pub fn get_split_index(&self) -> usize {
        self.split_feature[0].unwrap() as usize
    }

    #[allow(dead_code)]
    pub fn get_split_value(&self) -> f32 {
        self.threshold[0]
    }
}
