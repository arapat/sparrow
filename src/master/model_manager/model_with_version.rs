
use commons::INIT_MODEL_PREFIX;
use commons::Model;
use commons::tree::UpdateList;
use super::scheduler::kdtree::Grid;


pub struct ModelWithVersion {
    pub model: Model,
    model_prefix: String,
    gamma_version: usize,
    pub model_sig: String,
}


impl ModelWithVersion {
    pub fn new(model: Model) -> ModelWithVersion {
        let (model_prefix, gamma_version) = (INIT_MODEL_PREFIX.to_string(), 0);
        let model_sig = get_model_sig(&model_prefix, gamma_version);
        ModelWithVersion {
            model: model,
            model_prefix: model_prefix,
            gamma_version: gamma_version,
            model_sig: model_sig,
        }
    }

    pub fn add_grid(&mut self, grid: Grid) -> usize {
        let grid_index = self.model.add_grid(grid);
        self.model_prefix = format!("master_{}", self.model.tree_size);
        self.model_sig = get_model_sig(&self.model_prefix, self.gamma_version);
        grid_index
    }

    pub fn update(
        &mut self, patch: &UpdateList, new_prefix: &String, gamma: f32,
    ) -> (Vec<usize>, usize, usize) {
        let node_indices: Vec<(usize, bool)> = self.model.append_patch(&patch, gamma, true);
        let (count_new, count_updates) = patch.is_new.iter().fold(
            (0, 0), |(new, old), t| { if *t { (new + 1, old) } else { (new, old + 1) } });
        self.model_prefix = new_prefix.clone();
        self.model_sig = get_model_sig(&self.model_prefix, self.gamma_version);
        (node_indices.into_iter().map(|(i, _)| i).collect(), count_new, count_updates)
    }

    pub fn update_gamma(&mut self, gamma_version: usize) {
        self.gamma_version = gamma_version;
        self.model_sig = get_model_sig(&self.model_prefix, self.gamma_version);
    }

    pub fn print_log(&self) {
        let num_roots = self.model.depth.iter().filter(|t| **t == 1).count();
        debug!("model stats, status, {}, {}, {}, {}, {}, {}",
                self.model.tree_size,
                self.model.size(),
                num_roots,
                self.model_prefix,
                self.gamma_version,
                self.model_sig);
    }
}


fn get_model_sig(prefix: &String, gamma_version: usize) -> String {
    format!("{}_{}", prefix, gamma_version)
}