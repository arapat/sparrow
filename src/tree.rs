use rayon::prelude::*;
use std::ops::Range;
use std::collections::VecDeque;
use super::Example;
use super::TFeature;

use commons::is_zero;


// split_feature, threshold, evaluation
type Condition = (usize, TFeature, bool);

/*
Why JSON but not binary?
    - Readable for human
    - Compatible with Python
    - BufReader-friendly by using newline as separator
*/
#[derive(Serialize, Deserialize, Debug)]
pub struct Tree {
    pub tree_size:       usize,
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
    pub last_gamma:     f32,
    pub base_version:   usize,
    pub model_updates:  UpdateList,
}

impl Clone for Tree {
    fn clone(&self) -> Tree {
        Tree {
            tree_size:      self.tree_size,
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
            last_gamma:     self.last_gamma,
            base_version:   self.base_version,
            model_updates:  self.model_updates.clone(),
        }
    }
}

impl Tree {
    pub fn new(max_nodes: usize, base_pred: f32, base_gamma: f32) -> Tree {
        let mut tree = Tree {
            tree_size:      0,
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
            last_gamma:     0.0,
            base_version:   0,
            model_updates:  UpdateList::new(),
        };
        tree.add_node(0, 0, 0, false, base_pred, base_gamma);
        tree
    }

    pub fn size(&self) -> usize {
        self.model_updates.size
    }

    pub fn add_node(
        &mut self, parent: i32,
        feature: usize, threshold: TFeature, evaluation: bool, pred_value: f32, gamma: f32,
    ) -> Option<usize> {
        let depth = {
            if parent < 0 {
                0
            } else if parent == 0 {
                1
            } else {
                self.leaf_depth[parent as usize] + 1
            }
        };
        let node = self.find_child_node(parent, feature, threshold, evaluation);
        let parent = parent as usize;
        let (new_index, is_new) = {
            if let Some(index) = node {
                self.predicts[index] += pred_value;
                (index, false)
            } else {
                self.parent.push(parent);
                self.children.push(vec![]);
                self.split_feature.push(feature);
                self.threshold.push(threshold);
                self.evaluation.push(evaluation);
                self.predicts.push(pred_value);
                self.leaf_depth.push(depth);
                self.num_active.push(0);
                let index = self.tree_size;
                self.latest_child.push(index);
                if index > 0 {
                    self.children[parent].push(index);
                }
                self.tree_size += 1;
                (index, true)
            }
        };
        self.last_gamma = gamma;
        let condition = self.get_conditions(new_index);
        self.model_updates.add(parent, feature, threshold, evaluation, pred_value, condition);
        /*
        // No longer needed because the tree is not used for predicting during training
        let mut ancestor = index;
        while ancestor > 0 {
            ancestor = self.parent[ancestor];
            self.latest_child[ancestor] = index;
        }
        */
        debug!("new-tree-node, {}, {}, {}, {}, {}, {}, {}, {}",
               new_index, is_new, parent, depth, feature, threshold, evaluation, pred_value);
        if is_new {
            Some(new_index)
        } else {
            None
        }
    }

    fn find_child_node(
        &self, parent: i32, feature: usize, threshold: TFeature, evaluation: bool,
    ) -> Option<usize> {
        if parent < 0 {
            return Some(0);
        }
        let mut ret = None;
        if parent >= self.children.len() as i32 {
            return None;
        }
        self.children[parent as usize].iter().for_each(|index| {
            if self.split_feature[*index] == feature &&
                is_zero((self.threshold[*index] - threshold).into()) &&
                self.evaluation[*index] == evaluation {
                    ret = Some(*index);
            }
        });
        ret
    }

    /*
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
    */

    pub fn get_prediction_tree(&self, data: &Example) -> f32 {
        let feature = &(data.feature);
        let mut queue = VecDeque::new();
        queue.push_back(0);
        let mut prediction = 0.0;
        while !queue.is_empty() {
            let node = queue.pop_front().unwrap();
            prediction += self.predicts[node];
            self.children[node].iter().filter(|child| {
                (feature[self.split_feature[**child]] <= self.threshold[**child]) ==
                    self.evaluation[**child]
            }).for_each(|t| {
                queue.push_back(*t);
            });
        }
        prediction
    }

    pub fn get_prediction(&self, data: &Example, version: usize) -> (f32, (usize, usize)) {
        self.model_updates.get_prediction(data, version)
    }

    pub fn is_visited(&self, data: &Example, target: usize) -> bool {
        let feature = &(data.feature);
        let mut queue = VecDeque::new();
        queue.push_back(0);
        while !queue.is_empty() {
            let node = queue.pop_front().unwrap();
            if node == target {
                return true;
            }
            self.children[node].iter().filter(|child| {
                (feature[self.split_feature[**child]] <= self.threshold[**child]) ==
                    self.evaluation[**child]
            }).for_each(|t| {
                queue.push_back(*t);
            });
        }
        false
    }

    pub fn append_patch(
        &mut self, patch: &TreeSlice, last_gamma: f32, overwrite_root: bool,
    ) -> Vec<usize> {
        let mut i = {
            if overwrite_root {
                self.predicts[0] = patch.predicts[0];
                1
            } else {
                0
            }
        };
        let mut node_indices = vec![];
        while i < patch.size {
            node_indices.push(self.add_node(
                patch.parent[i] as i32, patch.split_feature[i], patch.threshold[i],
                patch.evaluation[i], patch.predicts[i], 0.0,
            ));
            i += 1;
        }
        self.last_gamma = last_gamma;
        self.base_version = self.model_updates.size;
        node_indices.iter()
                    .filter(|t| t.is_some())
                    .map(|t| self.leaf_depth[t.unwrap()])
                    .collect()
    }

    pub fn get_conditions(&self, node_index: usize) -> Vec<Condition> {
        let mut ret = vec![];
        // split_feature, threshold, evaluation
        let mut index = node_index;
        while index > 0 {
            ret.push((
                self.split_feature[index],
                self.threshold[index],
                self.evaluation[index],
            ));
            index = self.parent[index];
        }
        ret.reverse();
        ret
    }
}

impl PartialEq for Tree {
    fn eq(&self, other: &Tree) -> bool {
        let k = self.tree_size;
        if k == other.tree_size &&
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
    pub fn new(tree: &Tree, index_range: Range<usize>) -> TreeSlice {
        TreeSlice {
            size:          index_range.end - index_range.start,
            parent:        tree.model_updates.parent[index_range.clone()].to_vec(),
            split_feature: tree.model_updates.feature[index_range.clone()].to_vec(),
            threshold:     tree.model_updates.threshold[index_range.clone()].to_vec(),
            evaluation:    tree.model_updates.evaluation[index_range.clone()].to_vec(),
            predicts:      tree.model_updates.predicts[index_range.clone()].to_vec(),
        }
    }
}


#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateList {
    pub size:   usize,
    pub parent:     Vec<usize>,
    pub feature:    Vec<usize>,
    pub threshold:  Vec<TFeature>,
    pub evaluation: Vec<bool>,
    pub predicts:   Vec<f32>,
    condition:  Vec<Vec<Condition>>,
}


impl UpdateList {
    pub fn new() -> UpdateList {
        UpdateList {
            size: 0,
            parent:     vec![],
            feature:    vec![],
            threshold:  vec![],
            evaluation: vec![],
            predicts:   vec![],
            condition:  vec![],
        }
    }

    pub fn get_prediction(&self, data: &Example, version: usize) -> (f32, (usize, usize)) {
        let feature = &(data.feature);
        let pred: f32 = self.predicts[version..self.size].par_iter().zip(
            self.condition[version..self.size].par_iter()
        ).map(|(predict, conditions)| {
            let mut valid = true;
            for (split_feature, threshold, evaluation) in conditions {
                if (feature[*split_feature] <= *threshold) != *evaluation {
                    valid = false;
                    break;
                }
            }
            if valid {
                *predict
            } else {
                0.0
            }
        }).sum();
        (pred, (self.size, version))
    }

    pub fn add(
        &mut self,
        parent: usize,
        feature: usize,
        threshold: TFeature,
        evaluation: bool,
        predict: f32,
        conditions: Vec<Condition>,
    ) {
        self.parent.push(parent);
        self.feature.push(feature);
        self.threshold.push(threshold);
        self.evaluation.push(evaluation);
        self.predicts.push(predict);
        self.condition.push(conditions);
        self.size += 1;
    }
}


impl Clone for UpdateList {
    fn clone(&self) -> UpdateList {
        UpdateList {
            size:       self.size,
            parent:     self.parent.clone(),
            feature:    self.feature.clone(),
            threshold:  self.threshold.clone(),
            evaluation: self.evaluation.clone(),
            predicts:   self.predicts.clone(),
            condition:  self.condition.clone(),
        }
    }
}
