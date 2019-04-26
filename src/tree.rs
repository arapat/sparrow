use std::ops::Range;
use std::collections::VecDeque;
use super::Example;
use super::TFeature;

use commons::is_zero;


/*
Why JSON but not binary?
    - Readable for human
    - Compatible with Python
    - BufReader-friendly by using newline as separator
*/
#[derive(Serialize, Deserialize, Debug)]
pub struct Tree {
    pub size: usize,
    parent:         Vec<usize>,
    children:       Vec<Vec<usize>>,
    split_feature:  Vec<usize>,
    threshold:      Vec<TFeature>,
    evaluation:     Vec<bool>,
    predicts:       Vec<f32>,
    leaf_depth:     Vec<usize>,
    latest_child:   Vec<usize>,
    is_active:      Vec<bool>,
    num_active:     Vec<usize>,
}

impl Clone for Tree {
    fn clone(&self) -> Tree {
        Tree {
            size:     self.size,
            parent:         self.parent.clone(),
            children:       self.children.clone(),
            split_feature:  self.split_feature.clone(),
            threshold:      self.threshold.clone(),
            evaluation:     self.evaluation.clone(),
            predicts:       self.predicts.clone(),
            leaf_depth:     self.leaf_depth.clone(),
            latest_child:   self.latest_child.clone(),
            is_active:      self.is_active.clone(),
            num_active:     self.num_active.clone(),
        }
    }
}

impl Tree {
    pub fn new(max_nodes: usize, base_pred: f32) -> Tree {
        let mut tree = Tree {
            size:           0,
            parent:         Vec::with_capacity(max_nodes),
            children:       Vec::with_capacity(max_nodes),
            split_feature:  Vec::with_capacity(max_nodes),
            threshold:      Vec::with_capacity(max_nodes),
            evaluation:     Vec::with_capacity(max_nodes),
            predicts:       Vec::with_capacity(max_nodes),
            leaf_depth:     Vec::with_capacity(max_nodes),
            latest_child:   Vec::with_capacity(max_nodes),
            is_active:      Vec::with_capacity(max_nodes),
            num_active:     Vec::with_capacity(max_nodes),
        };
        tree.add_node(0, 0, 0, false, base_pred);
        tree
    }

    // TODO: replace split func with add_node
    pub fn add_node(
        &mut self, parent: usize,
        feature: usize, threshold: TFeature, evaluation: bool, pred_value: f32,
    ) -> usize {
        let index = self.size;
        let depth = {
            if parent == 0 {
                1
            } else {
                self.leaf_depth[parent] + 1
            }
        };
        self.parent.push(parent);
        self.children.push(vec![]);
        self.split_feature.push(feature);
        self.threshold.push(threshold);
        self.evaluation.push(evaluation);
        self.predicts.push(pred_value);
        self.leaf_depth.push(depth);
        self.is_active.push(false);
        self.num_active.push(0);
        self.latest_child.push(index);
        self.size += 1;
        if index > 0 {
            self.children[parent].push(index);
        }
        let mut ancestor = index;
        while ancestor > 0 {
            ancestor = self.parent[ancestor];
            self.latest_child[ancestor] = index;
        }
        debug!("new-tree-node, {}, {}, {}, {}, {}, {}, {}",
               index, parent, depth, feature, threshold, evaluation, pred_value);
        index
    }

    pub fn mark_active(&mut self, index: usize) {
        let mut ancestor = index;
        self.is_active[index] = true;
        while ancestor > 0 {
            self.num_active[ancestor] += 1;
            ancestor = self.parent[ancestor];
        }
        self.num_active[ancestor] += 1;
    }

    pub fn unmark_active(&mut self, index: usize) {
        let mut ancestor = index;
        self.is_active[index] = false;
        while ancestor > 0 {
            self.num_active[ancestor] -= 1;
            ancestor = self.parent[ancestor];
        }
        self.num_active[ancestor] -= 1;
    }

    pub fn get_prediction(&self, data: &Example, version: usize) -> (f32, (usize, usize)) {
        let feature = &(data.feature);
        let mut queue = VecDeque::new();
        queue.push_back(0);
        let mut prediction = 0.0;
        while !queue.is_empty() {
            let node = queue.pop_front().unwrap();
            if version <= node {
                prediction += self.predicts[node];
            }
            self.children[node].iter().filter(|child| {
                version <= self.latest_child[**child] && (
                    (feature[self.split_feature[**child]] <= self.threshold[**child]) ==
                        self.evaluation[**child]
                )
            }).for_each(|t| {
                queue.push_back(*t);
            });
        }
        (prediction, (self.latest_child[0] + 1, version))
    }

    pub fn get_active_nodes(&self, data: &Example) -> Vec<usize> {
        let feature = &(data.feature);
        let mut queue = VecDeque::new();
        queue.push_back(0);
        let mut active = vec![];
        while !queue.is_empty() {
            let node = queue.pop_front().unwrap();
            if self.is_active[node] {
                active.push(node);
            }
            self.children[node].iter().filter(|child| {
                self.num_active[**child] > 0 && (
                    (feature[self.split_feature[**child]] <= self.threshold[**child]) ==
                        self.evaluation[**child]
                )
            }).for_each(|t| {
                queue.push_back(*t);
            });
        }
        active
    }

    pub fn append_patch(&mut self, patch: &TreeSlice, overwrite_root: bool) {
        let mut i = {
            if overwrite_root {
                self.predicts[0] = patch.predicts[0];
                1
            } else {
                0
            }
        };
        while i < patch.size {
            self.add_node(
                patch.parent[i], patch.split_feature[i], patch.threshold[i],
                patch.evaluation[i], patch.predicts[i],
            );
            i += 1;
        }
    }
}

impl PartialEq for Tree {
    fn eq(&self, other: &Tree) -> bool {
        let k = self.size;
        if k == other.size &&
           self.split_feature[0..k] == other.split_feature[0..k] &&
           self.parent[0..k] == other.parent[0..k] &&
           self.children[0..k] == other.children[0..k] {
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


#[derive(Serialize, Deserialize, Debug)]
pub struct TreeSlice {
    pub size:           usize,
    pub parent:         Vec<usize>,
    pub split_feature:  Vec<usize>,
    pub threshold:      Vec<TFeature>,
    pub evaluation:     Vec<bool>,
    pub predicts:       Vec<f32>,
}


impl TreeSlice {
    pub fn new(tree: &Tree, range: Range<usize>) -> TreeSlice {
        TreeSlice {
            size:          range.end - range.start,
            parent:        tree.parent[range.clone()].to_vec(),
            split_feature: tree.split_feature[range.clone()].to_vec(),
            threshold:     tree.threshold[range.clone()].to_vec(),
            evaluation:    tree.evaluation[range.clone()].to_vec(),
            predicts:      tree.predicts[range.clone()].to_vec(),
        }
    }
}