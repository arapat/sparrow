use TFeature;
use commons::Example;

use commons::is_zero;

type DimScaleType = u16;


/*
Why JSON but not binary?
    - Readable for human
    - Compatible with Python
    - BufReader-friendly by using newline as separator
*/
#[derive(Serialize, Deserialize, Debug)]
pub struct Tree {
    max_leaves:     DimScaleType,
    pub num_leaves: DimScaleType,
    left_child:     Vec<DimScaleType>,
    right_child:    Vec<DimScaleType>,
    split_feature:  Vec<Option<DimScaleType>>,
    threshold:      Vec<TFeature>,
    predicts:       Vec<f32>,
    leaf_depth:     Vec<DimScaleType>
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
            predicts:       self.predicts.clone(),
            leaf_depth:     self.leaf_depth.clone()
        }
    }
}

impl Tree {
    pub fn new(num_splits: usize) -> Tree {
        let max_leaves = ((num_splits + 1) * 2) as u16;
        let mut tree = Tree {
            max_leaves:     max_leaves,
            num_leaves:     0,
            left_child:     Vec::with_capacity(max_nodes as usize),
            right_child:    Vec::with_capacity(max_nodes as usize),
            split_feature:  Vec::with_capacity(max_nodes as usize),
            threshold:      Vec::with_capacity(max_nodes as usize),
            predicts:       Vec::with_capacity(max_nodes as usize),
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
        self.predicts.shrink_to_fit();
        self.leaf_depth.shrink_to_fit();
    }

    pub fn split(
        &mut self, parent: usize, feature: usize, threshold: TFeature,
        left_predict: f32, right_predict: f32,
    ) -> (u16, u16) {
        let predict = self.predicts[parent];
        let parent_depth = self.leaf_depth[parent];

        self.split_feature[parent] = Some(feature as DimScaleType);
        self.threshold[parent] = threshold;
        self.left_child[parent] = self.num_leaves as DimScaleType;
        self.add_new_node(predict + left_predict, parent_depth + 1);
        self.right_child[parent] = self.num_leaves as DimScaleType;
        self.add_new_node(predict + right_predict, parent_depth + 1);
        (self.left_child[parent], self.right_child[parent])
    }

    pub fn get_leaf_index_prediction(&self, data: &Example) -> (usize, f32) {
        let mut node: usize = 0;
        let feature = &(data.feature);
        while let Some(split_feature) = self.split_feature[node] {
            node = if feature[split_feature as usize] <= self.threshold[node] {
                self.left_child[node]
            } else {
                self.right_child[node]
            } as usize;
        }
        (node, self.predicts[node])
    }

    pub fn get_leaf_prediction(&self, data: &Example) -> f32 {
        self.get_leaf_index_prediction(data).1
    }

    pub fn is_full_tree(&self) -> bool {
        self.num_leaves >= self.max_leaves
    }

    fn add_new_node(&mut self, predict: f32, depth: DimScaleType) {
        self.num_leaves += 1;
        self.left_child.push(0);
        self.right_child.push(0);
        self.split_feature.push(None);
        self.threshold.push(0);
        self.predicts.push(predict);
        self.leaf_depth.push(depth);
    }
}

impl PartialEq for Tree {
    fn eq(&self, other: &Tree) -> bool {
        let k = self.num_leaves;
        if k == other.num_leaves &&
           self.split_feature[0..k] == other.split_feature[0..k] &&
           self.left_child[0..k] == other.left_child[0..k] &&
           self.right_child[0..k] == other.right_child[0..k] {
               for i in 0..k {
                   if self.threshold[i] != other.threshold[i] ||
                      !is_zero(self.predicts[i] - other.predicts[i]) {
                          return false;
                      }
               }
               return true;
        }
        false
    }
}


impl Eq for Tree {}