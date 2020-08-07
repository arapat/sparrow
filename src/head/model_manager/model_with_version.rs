
use commons::Model;
use commons::tree::UpdateList;
use head::scheduler::kdtree::Grid;


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

    pub fn add_grid(&mut self, grid: Grid) -> usize {
        self.model.add_grid(grid)
    }

    pub fn update(
        &mut self, patch: &UpdateList, last_update_from: &String,
    ) -> (Vec<usize>, usize, usize) {
        let node_indices: Vec<(usize, bool)> = self.model.append_patch(&patch, true);
        let (count_new, count_updates) = patch.is_new.iter().fold(
            (0, 0), |(new, old), t| { if *t { (new + 1, old) } else { (new, old + 1) } });
        self.last_update_from = last_update_from.clone();
        self.model_size = self.model_size;
        (node_indices.into_iter().map(|(i, _)| i).collect(), count_new, count_updates)
    }

    pub fn update_gamma(&mut self, gamma_version: usize) {
        self.gamma_version = gamma_version;
    }

    pub fn print_log(&self) {
        let num_roots = self.model.depth.iter().filter(|t| **t == 1).count();
        debug!("model stats, status, {}, {}, {}, {}",
                self.model.tree_size,
                self.model.size(),
                num_roots,
                self.gamma_version);
    }
}


fn get_model_sig(prefix: &String, gamma_version: usize) -> String {
    format!("{}_{}", prefix, gamma_version)
}