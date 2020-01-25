use rayon::prelude::*;
use std::ops::Range;
use std::collections::VecDeque;

use Example;
use TFeature;
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
    pub parent:         Vec<usize>,
    pub children:       Vec<Vec<usize>>,
    split_feature:  Vec<usize>,
    threshold:      Vec<TFeature>,
    evaluation:     Vec<bool>,
    predicts:       Vec<f32>,
    is_active:      Vec<bool>,
    pub depth:      Vec<usize>,
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
            depth:          self.depth.clone(),
            is_active:      self.is_active.clone(),
            last_gamma:     self.last_gamma,
            base_version:   self.base_version,
            model_updates:  self.model_updates.clone(),
        }
    }
}

// TODO: remove the queue-like accessing if we no longer use the AD-tree structure
impl Tree {
    pub fn new(max_nodes: usize) -> Tree {
        Tree {
            tree_size:      0,
            parent:         Vec::with_capacity(max_nodes),
            children:       Vec::with_capacity(max_nodes),
            split_feature:  Vec::with_capacity(max_nodes),
            threshold:      Vec::with_capacity(max_nodes),
            evaluation:     Vec::with_capacity(max_nodes),
            predicts:       Vec::with_capacity(max_nodes),
            depth:          Vec::with_capacity(max_nodes),
            is_active:      Vec::with_capacity(max_nodes),
            last_gamma:     0.0,
            base_version:   0,
            model_updates:  UpdateList::new(),
        }
    }

    pub fn size(&self) -> usize {
        self.model_updates.size
    }

    pub fn add_nodes(
        &mut self, parent: usize,
        feature: usize, threshold: TFeature, pred_value: (f32, f32), gamma: f32,
    ) -> (usize, usize) {
        // TODO: set always add new a parameter
        let always_new_node = true;
        (
            self.add_node(parent, feature, threshold, true, pred_value.0, gamma, always_new_node),
            self.add_node(parent, feature, threshold, false, pred_value.1, gamma, always_new_node)
        )
    }

    // TODO: allow update root
    pub fn add_root(&mut self, pred_value: f32, gamma: f32) -> usize {
        assert_eq!(self.tree_size, 0);

        self.parent.push(0);
        self.children.push(vec![]);
        self.split_feature.push(0);
        self.threshold.push(0 as TFeature);
        self.evaluation.push(false);
        self.predicts.push(pred_value);
        self.depth.push(0);
        self.tree_size += 1;

        self.last_gamma = gamma;
        self.model_updates.add(-1, 0, 0, false, pred_value, vec![], true);

        debug!("new-tree-node, 0, true, 0, 0, 0, 0, false, {}", pred_value);
        0
    }

    fn add_node(
        &mut self, parent: usize,
        feature: usize, threshold: TFeature, evaluation: bool, pred_value: f32, gamma: f32,
        always_new_node: bool,
    ) -> usize {
        let depth = self.depth[parent] + 1;
        let node =
            if always_new_node { None } else {
                self.find_child_node(parent, feature, threshold, evaluation)
            };
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
                self.depth.push(depth);
                let index = self.tree_size;
                self.children[parent].push(index);
                self.tree_size += 1;
                (index, true)
            }
        };
        self.last_gamma = gamma;
        let condition = self.get_conditions(new_index);
        self.model_updates.add(
            parent as i32, feature, threshold, evaluation, pred_value, condition, is_new);
        debug!("new-tree-node, {}, {}, {}, {}, {}, {}, {}, {}",
               new_index, is_new, parent, depth, feature, threshold, evaluation, pred_value);
        new_index
    }

    fn find_child_node(
        &self, parent: usize, feature: usize, threshold: TFeature, evaluation: bool,
    ) -> Option<usize> {
        let mut ret = None;
        self.children[parent].iter().for_each(|index| {
            if self.split_feature[*index] == feature &&
                is_zero((self.threshold[*index] - threshold).into()) &&
                self.evaluation[*index] == evaluation {
                    ret = Some(*index);
            }
        });
        ret
    }

    #[allow(dead_code)]
    pub fn get_prediction_tree(&self, data: &Example) -> f32 {
        if self.tree_size <= 0 {
            return 0.0;
        }
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

    pub fn visit_tree(&self, data: &Example, counter: &mut Vec<u32>) {
        if self.tree_size <= 0 {
            return;
        }
        let feature = &(data.feature);
        let mut queue = VecDeque::new();
        queue.push_back(0);
        while !queue.is_empty() {
            let node = queue.pop_front().unwrap();
            counter[node] += 1;
            self.children[node].iter().filter(|child| {
                (feature[self.split_feature[**child]] <= self.threshold[**child]) ==
                    self.evaluation[**child]
            }).for_each(|t| {
                queue.push_back(*t);
            });
        }
    }

    pub fn get_prediction(&self, data: &Example, version: usize) -> (f32, (usize, usize)) {
        self.model_updates.get_prediction_ul(data, version)
    }

    pub fn is_visited(&self, data: &Example, target: usize) -> bool {
        if self.tree_size <= 0 {
            return false;
        }
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

    pub fn get_leaf_index_prediction(&self, starting_index: usize, data: &Example) -> usize {
        let mut node: usize = starting_index;
        let feature = &(data.feature);
        while self.children[node].len() > 0 {
            for child in self.children[node] {
                if (feature[self.split_feature[child]] <= self.threshold[child]) ==
                    self.evaluation[child] {
                        node = child;
                        break;
                    }
            }
        }
        node
    }

    // return the indices of the added nodes and whether they are new nodes
    pub fn append_patch(
        &mut self, patch: &UpdateList, last_gamma: f32, always_new_node: bool,
    ) -> Vec<(usize, bool)> {
        let prev_tree_size = self.tree_size;
        let mut node_indices = vec![];
        for i in 0..patch.size {
            if patch.parent[i] < 0 {
                node_indices.push(self.add_root(patch.predicts[i], 0.0));
            } else {
                node_indices.push(self.add_node(
                    patch.parent[i] as usize, patch.feature[i], patch.threshold[i],
                    patch.evaluation[i], patch.predicts[i], 0.0, always_new_node,
                ));
            }
        }
        self.last_gamma = last_gamma;
        self.base_version = self.model_updates.size;
        node_indices.iter()
                    .map(|t| (*t, (*t) >= prev_tree_size))  // newly added indices
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
pub struct UpdateList {
    pub size:   usize,
    pub parent:     Vec<i32>,
    pub feature:    Vec<usize>,
    pub threshold:  Vec<TFeature>,
    pub evaluation: Vec<bool>,
    pub predicts:   Vec<f32>,
    condition:  Vec<Vec<Condition>>,
    // debug info
    pub is_new:     Vec<bool>,
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
            is_new:     vec![],
        }
    }

    pub fn get_prediction_ul(&self, data: &Example, version: usize) -> (f32, (usize, usize)) {
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
        parent: i32,
        feature: usize,
        threshold: TFeature,
        evaluation: bool,
        predict: f32,
        conditions: Vec<Condition>,
        is_new: bool,
    ) {
        self.parent.push(parent);
        self.feature.push(feature);
        self.threshold.push(threshold);
        self.evaluation.push(evaluation);
        self.predicts.push(predict);
        self.condition.push(conditions);
        self.is_new.push(is_new);
        self.size += 1;
    }

    pub fn create_slice(&self, index_range: Range<usize>) -> UpdateList {
        UpdateList {
            size:          index_range.end - index_range.start,
            parent:        self.parent[index_range.clone()].to_vec(),
            feature:       self.feature[index_range.clone()].to_vec(),
            threshold:     self.threshold[index_range.clone()].to_vec(),
            evaluation:    self.evaluation[index_range.clone()].to_vec(),
            predicts:      self.predicts[index_range.clone()].to_vec(),
            condition:     vec![],
            is_new:        self.is_new[index_range.clone()].to_vec(),
        }
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
            is_new:     self.is_new.clone(),
        }
    }
}
