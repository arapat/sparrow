
use commons::model::Model;
use commons::tree::Tree;


#[derive(Clone)]
pub struct ModelWithVersion {
    pub model: Model,
    pub last_update_from: String,
    pub model_size: usize,
    gamma_version: usize,
}


impl ModelWithVersion {
    pub fn new(model: Model, last_update_from: String) -> ModelWithVersion {
        ModelWithVersion {
            last_update_from: last_update_from,
            model_size: model.size(),
            gamma_version: 0,

            model: model,
        }
    }

    pub fn update(&mut self, patch: Tree, last_update_from: &String) {
        self.model.append(patch);
        self.last_update_from = last_update_from.clone();
        self.model_size = self.model.size();
    }

    pub fn update_gamma(&mut self, gamma_version: usize) {
        self.gamma_version = gamma_version;
    }

    pub fn print_log(&self) {
        debug!("model stats, status, {}, {}", self.model.size(), self.gamma_version);
    }

    pub fn size(&self) -> usize {
        self.model.size()
    }
}