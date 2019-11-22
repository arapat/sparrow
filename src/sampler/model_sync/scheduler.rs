use std::cmp::max;
use std::cmp::min;

use commons::packet::Packet;
use commons::persistent_io::upload_assignments;
use super::Gamma;
use super::ModelStats;

// (last_failed_gamma, current_scanner_id)
type NodeStatus = (f32, Option<usize>);


const MAX_EMPTY_TREE: usize = 10;


pub struct Scheduler {
    node_status: Vec<NodeStatus>,
    scanner_task: Vec<Option<usize>>,
    exp_name: String,
}


impl Scheduler {
    pub fn new(
        num_machines: usize, model_stats: &ModelStats, exp_name: &String, gamma: &Gamma,
    ) -> Scheduler {
        let mut scheduler = Scheduler {
            node_status: vec![(1.0, None); model_stats.model.tree_size],
            scanner_task: vec![None; num_machines],
            exp_name: exp_name.clone(),
        };
        // scheduler.scanner_task[0] = Some(0);
        // if scheduler.node_status.len() > 0 {
        //     scheduler.node_status[0] = (1.0, Some(0));
        // }
        // upload_assignments(&scheduler.scanner_task, exp_name);
        scheduler.update(model_stats, gamma);
        scheduler
    }

    pub fn update(&mut self, model_stats: &ModelStats, gamma: &Gamma) -> (usize, bool) {
        // if there is no enough root nodes, set the cluster into the emergency state
        if model_stats.avail_new_tree <= self.node_status.len() {
            let mut num_updates = 0;
            for scanner_task in self.scanner_task.iter_mut() {
                if scanner_task.is_none() {
                    *scanner_task = Some(0);
                    num_updates += 1;
                } else if *scanner_task != Some(0) {
                    let node_id = scanner_task.take().unwrap();
                    let (last_failed_gamma, _) = self.node_status[node_id];
                    self.node_status[node_id] = (last_failed_gamma, None);
                    *scanner_task = Some(0);
                    num_updates += 1;
                }
            }
            if num_updates > 0 {
                debug!("model-manager, assign all set to root, {}", num_updates);
                upload_assignments(&self.scanner_task, &self.exp_name);
            }
            return (num_updates, false);
        }

        let num_nodes = self.node_status.len();
        let valid_nodes: Vec<usize> = (0..num_nodes).filter(|node_index| {
            self.is_valid_node(*node_index, model_stats, gamma)
        }).collect();
        let idle_scanners: Vec<usize> =
            self.scanner_task.iter()
                .enumerate()
                .filter(|(_, assignment)| assignment.is_none())
                .map(|(scanner_index, _)| scanner_index)
                .collect();

        let num_idle_scanners = idle_scanners.len();
        let num_updates = min(num_idle_scanners, valid_nodes.len());
        for (scanner_index, node_index) in idle_scanners.into_iter().zip(valid_nodes.into_iter()) {
            let (last_failed_gamma, _) = self.node_status[node_index];
            self.node_status[node_index] = (last_failed_gamma, Some(scanner_index));
            self.scanner_task[scanner_index] = Some(node_index);
            let gamma_val = if node_index == 0 { gamma.root_gamma } else { gamma.gamma };
            debug!("model-manager, assign, {}, {}, {}, {}",
                    scanner_index, node_index, last_failed_gamma, gamma_val);
        }
        if num_updates > 0 {
            debug!("model-manager, assign updates, {}", num_updates);
            upload_assignments(&self.scanner_task, &self.exp_name);
            (num_updates, false)
        } else if num_idle_scanners > 0 {
            let restricted_by_gamma = (1..num_nodes).filter(|node_index| {
                self.restricted_by_gamma(*node_index, model_stats, gamma)
            }).count();
            (0, restricted_by_gamma > 0)
        } else {
            (0, false)
        }
    }

    pub fn handle_success(
        &mut self, packet: &Packet, model_stats: &ModelStats, gamma: &Gamma, node_count: u32,
    ) -> bool {
        let node_id = self.get_node_id(packet, "success");
        if node_id.is_none() {
            return false;
        }
        let node_id = node_id.unwrap();
        if !self.is_valid_node(node_id, model_stats, gamma) {
            let (last_failed_gamma, _) = self.node_status[node_id];
            self.node_status[node_id] = (last_failed_gamma, None);
            self.scanner_task[packet.source_machine_id] = None;
            let gamma_val = if node_id == 0 { gamma.root_gamma } else { gamma.gamma };
            debug!("model_manager, scheduler, success, reset, {}, {}, {}, {}, {}",
                    packet.source_machine_id, node_id, node_count, packet.gamma, gamma_val);
            true
        } else {
            false
        }
    }

    pub fn handle_failure(&mut self, packet: &Packet, gamma: &Gamma, node_count: u32) {
        if let Some(node_id) = self.get_node_id(packet, "empty") {
            self.node_status[node_id] = (packet.gamma, None);
            self.scanner_task[packet.source_machine_id] = None;
            let gamma_val = if node_id == 0 { gamma.root_gamma } else { gamma.gamma };
            debug!("model_manager, scheduler, empty, reset, {}, {}, {}, {}, {}",
                    packet.source_machine_id, node_id, node_count, packet.gamma, gamma_val);
        }
    }

    pub fn append_new_nodes(&mut self, new_node_indices: &Vec<usize>) {
        let max_new_index = new_node_indices.iter().fold(0, |curr_max, val| max(curr_max, *val));
        let num_new_nodes = (max_new_index + 1) - self.node_status.len();
        self.node_status.append(&mut vec![(1.0, None); num_new_nodes]);
    }

    fn is_valid_node(&self, index: usize, model_stats: &ModelStats, gamma: &Gamma) -> bool {
        let (last_failed_gamma, assigner) = self.node_status[index];
        let num_child = model_stats.model.children[index].len();
        let depth = model_stats.model.depth[index];
        if index == 0 {
            // `depth < max_depth` and `old_gamma > root_gamma` is guaranteed for the root
            assigner.is_none() &&
                model_stats.avail_new_tree < max(self.scanner_task.len() * 2, MAX_EMPTY_TREE) &&
                num_child < model_stats.max_num_trees
        } else {
            assigner.is_none() && depth < model_stats.max_depth && num_child <= 0 &&
                last_failed_gamma > gamma.gamma
        }
    }

    fn restricted_by_gamma(&self, index: usize, model_stats: &ModelStats, gamma: &Gamma) -> bool {
        if index == 0 {
            return false;
        }
        let (last_failed_gamma, assigner) = self.node_status[index];
        let num_child = model_stats.model.children[index].len();
        let depth = model_stats.model.depth[index];
        assigner.is_none() && depth < model_stats.max_depth && num_child <= 0 &&
            last_failed_gamma <= gamma.gamma  // gamma is the only condition that does not pass
    }

    fn get_node_id(&self, packet: &Packet, desc: &str) -> Option<usize> {
        if self.scanner_task[packet.source_machine_id].is_none() {
            debug!("model_manager, scheduler, {}, no assignment, {}, {}, {}",
                    desc, packet.packet_signature, packet.source_machine, packet.source_machine_id);
            return None;
        }
        let node_id = self.scanner_task[packet.source_machine_id].unwrap();
        if node_id != packet.node_id {
            debug!("model_manager, scheduler, {}, node_id mismatch, {}, {}, {}, {}, {}",
                    desc, packet.packet_signature, packet.source_machine, packet.source_machine_id,
                    node_id, packet.node_id);
            return None;
        }
        Some(node_id)
    }

    pub fn print_log(&self, num_consecutive_err: u32, model_stats: &ModelStats, gamma: &Gamma) {
        let scanner_id: Vec<Option<_>> = self.node_status.iter().map(|t| t.1).collect();
        let last_failed_gamma: Vec<_> = self.node_status.iter().map(|t| t.0.to_string()).collect();
        let depth: Vec<_> = self.node_status.iter().enumerate()
                                .map(|(index, _)| model_stats.model.depth[index].to_string())
                                .collect();
        let num_child: Vec<_> = self.node_status.iter().enumerate()
                                    .map(|(index, _)| model_stats.model.children[index].len()
                                                                                       .to_string())
                                    .collect();
        debug!("model_manager, scheduler, status, {}, {}, {}, \
                |||, {}, |||, {}, |||, {}, |||, {}, |||, {}",
                num_consecutive_err, gamma.gamma, gamma.root_gamma,
                vec_to_string(&self.scanner_task), vec_to_string(&scanner_id),
                last_failed_gamma.join(", "), depth.join(", "), num_child.join(", "));
    }
}


fn vec_to_string(vec: &Vec<Option<usize>>) -> String {
    let s: Vec<String> = vec.iter().map(|t| {
        if t.is_none() {
            "nan".to_string()
        } else {
            t.unwrap().to_string()
        }
    }).collect();
    s.join(", ")
}

/*
if packet_stats.curr_nonroot_condition == UpdateSpeed::TooFast {
    for node_id in 0..node_status.len() {
        let (status, _remote_gamma, assignment) = node_status[node_id];
        node_status[node_id] = (status, 1.0, assignment);
    }
}
*/
