use Example;
use TFeature;

pub type Grid = Vec<(usize, TFeature, bool)>;
pub type Grids = Vec<Grid>;


pub struct KdTree {
    // node
    left: Option<Box<KdTree>>,
    right: Option<Box<KdTree>>,
    // common
    num_features: usize,
    // sum_weights: f32,
    // stem
    split_value: TFeature,
    split_dimension: usize,
    // leaf
    examples: Vec<Example>,
}

impl KdTree {
    pub fn new(examples: Vec<Example>, min_size: usize) -> Self {
        let mut tree = KdTree::empty(examples[0].feature.len());
        tree.examples = examples;
        tree.create_tree(min_size);
        tree
    }

    fn empty(num_features: usize) -> Self {
        KdTree {
            left: None,
            right: None,
            num_features: num_features,
            split_value: 0 as TFeature,
            split_dimension: 0,
            examples: vec![],
        }
    }

    pub fn get_leaves(&mut self) -> Grids {
        if self.left.is_none() {
            vec![vec![]]
        } else {
            let split_dimension = self.split_dimension;
            let split_value = self.split_value;
            let mut left_leaf  = self.left.take().unwrap().get_leaves();
            let mut right_leaf = self.right.take().unwrap().get_leaves();
            for leaf in &mut left_leaf {
                leaf.push((split_dimension, split_value, true));
            }
            for leaf in &mut right_leaf {
                leaf.push((split_dimension, split_value, false));
            }
            left_leaf.append(&mut right_leaf);
            left_leaf
        }
    }

    fn create_tree(&mut self, min_size: usize) {
        if self.examples.len() <= min_size * 2 {
            return;  // cannot further split because of the insufficient number of examples
        }

        let dim = rand::random::<usize>() % self.num_features;
        let mut values: Vec<TFeature> = self.examples.iter().map(|t| t.feature[dim]).collect();

        self.split_dimension = dim;
        self.split_value = get_median(&mut values);

        let mut left = Box::new(KdTree::empty(self.num_features));
        let mut right = Box::new(KdTree::empty(self.num_features));
        while !self.examples.is_empty() {
            let example = self.examples.swap_remove(0);
            if self.belongs_in_left(&example) {
                left.add_to_bucket(example)
            } else {
                right.add_to_bucket(example)
            }
        }
        left.create_tree(min_size);
        self.left = Some(left);
        right.create_tree(min_size);
        self.right = Some(right);
    }

    fn add_to_bucket(&mut self, example: Example) {
        self.examples.push(example);
    }

    fn belongs_in_left(&self, example: &Example) -> bool {
        example.feature[self.split_dimension] <= self.split_value
    }

}


fn get_median(numbers: &mut Vec<TFeature>) -> TFeature {
    let mid = numbers.len() / 2;
    numbers.sort();
    if numbers.len() % 2 == 0 {
        (numbers[mid - 1] as f32 / 2.0 + numbers[mid] as f32 / 2.0) as TFeature
    } else {
        numbers[mid]
    }
}


/*
#[cfg(test)]
mod tests {
    use Example;
    use commons::test_helper::get_n_random_examples;
    use master::model_manager::scheduler::kdtree::KdTree;

    #[test]
    fn test_kdtree() {
        let examples: Vec<Example> = get_n_random_examples(1000, 10).into_iter()
                                                                    .map(|(example, _)| { example })
                                                                    .collect();
        let min_size = 10;
        let mut kdtree = KdTree::new(examples.clone(), min_size);

        let grids = kdtree.get_leaves();
        let mut counts: Vec<usize> = vec![0; grids.len()];

        for example in examples {
            let mut match_index = None;
            for i in 0..grids.len() {
                let grid = &grids[i];
                let mut valid = true;
                for (index, thr, cond) in grid {
                    if (example.feature[*index as usize] <= *thr) != *cond {
                        valid = false;
                        break;
                    }
                }
                if valid {
                    assert!(match_index.is_none());
                    match_index = Some(i);
                }
            }
            assert!(match_index.is_some());
            counts[match_index.unwrap()] += 1;
        }

        for count in counts {
            assert!(min_size <= count && count <= min_size * 2);
        }
    }
}
*/