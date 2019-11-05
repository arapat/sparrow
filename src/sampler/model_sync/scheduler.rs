use std::cmp::min;

use commons::packet::Packet;
use commons::persistent_io::upload_assignments;
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
    pub fn new(num_machines: usize, model_stats: &ModelStats, exp_name: &String) -> Scheduler {
        let mut scheduler = Scheduler {
            node_status: vec![(1.0, None); model_stats.model.tree_size],
            scanner_task: vec![None; num_machines],
            exp_name: exp_name.clone(),
        };
        scheduler.update(model_stats, 0.9);
        scheduler
    }

    pub fn update(&mut self, model_stats: &ModelStats, gamma: f32) -> usize {
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

        let num_updates = min(idle_scanners.len(), valid_nodes.len());
        for (scanner_index, node_index) in idle_scanners.into_iter().zip(valid_nodes.into_iter()) {
            let (last_failed_gamma, _) = self.node_status[node_index];
            self.node_status[node_index] = (last_failed_gamma, Some(scanner_index));
            self.scanner_task[scanner_index] = Some(node_index);
            debug!("model-manager, assign, {}, {}", scanner_index, node_index);
        }
        if num_updates > 0 {
            debug!("model-manager, assign updates, {}", num_updates);
            upload_assignments(&self.scanner_task, &self.exp_name);
        }
        num_updates
    }

    pub fn handle_success(
        &mut self, packet: &Packet, model_stats: &ModelStats, node_count: u32,
    ) -> bool {
        let node_id = self.get_node_id(packet, "success");
        if node_id.is_none() {
            return false;
        }
        let node_id = node_id.unwrap();
        if !self.is_valid_node(node_id, model_stats, packet.gamma) {
            let (last_failed_gamma, _) = self.node_status[node_id];
            self.node_status[node_id] = (last_failed_gamma, None);
            self.scanner_task[packet.source_machine_id] = None;
            debug!("model_manager, scheduler, success, reset, {}, {}, {}, {}",
                    packet.source_machine_id, node_id, node_count, packet.gamma);
            true
        } else {
            false
        }
    }

    pub fn handle_failure(&mut self, packet: &Packet, node_count: u32) {
        if let Some(node_id) = self.get_node_id(packet, "empty") {
            self.node_status[node_id] = (packet.gamma, None);
            self.scanner_task[packet.source_machine_id] = None;
            debug!("model_manager, scheduler, empty, reset, {}, {}, {}, {}",
                    packet.source_machine_id, node_id, node_count, packet.gamma);
        }
    }

    pub fn append_new_nodes(&mut self, num_new_nodes: usize) {
        self.node_status.append(&mut vec![(1.0, None); num_new_nodes]);
    }

    fn is_valid_node(&self, index: usize, model_stats: &ModelStats, gamma: f32) -> bool {
        let (last_failed_gamma, assigner) = self.node_status[index];
        let num_child = model_stats.model.children[index].len();
        let depth = model_stats.model.depth[index];
        if index == 0 {
            // `depth < max_depth` and `old_gamma > root_gamma` is guaranteed for the root
            assigner.is_none() && model_stats.avail_new_tree < MAX_EMPTY_TREE &&
                num_child < model_stats.max_num_trees
        } else {
            assigner.is_none() && depth < model_stats.max_depth && num_child <= 0 &&
                last_failed_gamma > gamma
        }
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

    pub fn print_log(&self, num_consecutive_err: u32) {
        let node_status: Vec<Option<_>> = self.node_status.iter().map(|t| t.1).collect();
        debug!("model_manager, scheduler, status, {}, {}, {}",
                num_consecutive_err,
                vec_to_string(&self.scanner_task), vec_to_string(&node_status));
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
