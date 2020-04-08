use std::collections::BinaryHeap;

use Example;
use TFeature;

use crate::heap_element::HeapElement;
use crate::util;


pub struct KdTree {
    // node
    left: Option<Box<KdTree>>,
    right: Option<Box<KdTree>>,
    // common
    num_features: usize,
    size: usize,
    sum_weights: f32,
    // stem
    split_value: Option<TFeature>,
    split_dimension: Option<usize>,
    // leaf
    examples: Vec<Example>,
}

impl KdTree
    pub fn new(num_features: usize) -> Self {
        KdTree {
            left: None,
            right: None,
            num_features: num_features,
            size: 0,
            sum_weights: 0.0,
            split_value: None,
            split_dimension: None,
            examples: vec![],
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn create_tree(&mut self, min_size: usize) {
        if self.size() <= min_size * 2 {
            return;  // cannot further split because of the insufficient number of examples
        }

        let dim = rand::random::<usize>() % self.num_features;
        let values: Vec<TFeature> = examples.iter().map(|t| t.feature[dim]).collect();

        self.split_dimension = dim;
        self.split_value = get_median(&mut values);

        let mut left = Box::new(KdTree::new(self.num_features));
        let mut right = Box::new(KdTree::new(self.num_features));
        while !examples.is_empty() {
            let example = examples.swap_remove(0);
            if self.belongs_in_left(example.as_ref()) {
                left.add_to_bucket(point, data);
            } else {
                right.add_to_bucket(point, data);
            }
        }
        left.create_tree();
        self.left = Some(left);
        right.create_tree();
        self.right = Some(right);
    }

    pub fn batch_add_to_bucket(&mut self, examples: Vec<Examples>) {
        self.size = examples.len();
        self.examples = Some(examples);
    }

    pub fn get_leaves(&self) -> Vec<Vec<(usize, TFeature, bool)>> {
        if self.leaf.is_none() {
            vec![]
        } else {
            let split_dimension = self.split_dimension.clone().unwrap();
            let split_value = self.split_value.clone().unwrap();
            let mut left_leaf  = self.left.take().unwrap().get_leaves();
            let mut right_leaf = self.right.take().unwrap().get_leaves();
            left_leaf.push((split_dimension, split_value, true));
            right_leaf.push((split_dimension, split_value, true));
            left_leaf.append(&mut right_leaf);
            left_leaf
        }
    }

    fn add_to_bucket(&mut self, example: Example) {
        let mut examples = self.examples.take().unwrap();
        examples.push(example);
        self.size += 1;
        self.examples = Some(examples);
    }

    fn belongs_in_left(&self, example: &Example) -> bool {
        example.feature[self.split_dimension.unwrap()] <= self.split_value.unwrap()
    }

}


fn get_median(numbers: &mut Vec<TFeature>) -> usize {
    numbers.sort();
    let mid = numbers.len() / 2;
    if numbers.len() % 2 == 0 {
        (numbers[mid - 1] + numbers[mid]]) / 2.0
    } else {
        numbers[mid]
    }
}
