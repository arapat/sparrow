use rayon::prelude::*;
use commons::tree::Tree;
use Example;


#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Model {
    models: Vec<Tree>,
    pub base_size: usize,
}


impl Model {
    pub fn new() -> Model {
        Model {
            models: vec![],
            base_size: 0,
        }
    }

    pub fn size(&self) -> usize {
        self.models.len()
    }

    pub fn get_prediction(&self, data: &Example, version: usize) -> (f32, (usize, usize)) {
        let pred: f32 = self.models[version..self.size()]
                            .par_iter()
                            .map(|model| model.get_leaf_prediction(data))
                            .sum();
        (pred, (self.size(), version))
    }

    pub fn append(&mut self, update_tree: Tree) {
        self.models.push(update_tree);
    }

    pub fn get_last_new_tree(&self) -> Option<Tree> {
        if self.size() <= self.base_size {
            None
        } else {
            Some(self.models.last().unwrap().clone())
        }
    }

    pub fn set_base_size(&mut self) {
        self.base_size = self.size();
    }
}