mod grid;

use commons::bins::Bins;
use commons::packet::Packet;
use commons::persistent_io::upload_assignments;
use super::Gamma;
use self::grid::Grid;

pub struct Scheduler {
    grid: Grid,
    scanner_task: Vec<Option<(String, usize)>>,
    exp_name: String,
}


impl Scheduler {
    pub fn new(num_machines: usize, exp_name: &String, bins: &Vec<Bins>) -> Scheduler {
        let mut scheduler = Scheduler {
            grid: Grid::new(2, num_machines, bins),
            scanner_task: vec![None; num_machines],
            exp_name: exp_name.clone(),
        };
        scheduler.update();
        scheduler
    }

    pub fn update(&mut self) -> usize {
        let idle_scanners: Vec<usize> =
            self.scanner_task.iter()
                .enumerate()
                .filter(|(_, assignment)| assignment.is_none())
                .map(|(scanner_index, _)| scanner_index)
                .collect();
        let num_updates = self.assign(idle_scanners);
        if num_updates > 0 {
            debug!("model-manager, assign updates, {}", num_updates);
            upload_assignments(&self.scanner_task, &self.exp_name);
        }
        num_updates
    }

    // assign a non-taken grid to each idle scanner
    fn assign(&mut self, idle_scanners: Vec<usize>) -> usize {
        let assignment: Vec<(usize, (String, Vec<(usize, usize, usize)>))> =
            idle_scanners.into_iter()
                         .map(|scanner_id| (scanner_id, self.grid.get_new_grid(scanner_id)))
                         .collect();
        assignment.iter().for_each(|(scanner_id, (key, grid))| {
            // TODO: create this func
            let node_index = tree.create_node(grid);
            self.scanner_task[scanner_id] = Some((key, node_index));
            debug!("model-manager, assign, {}, {}", scanner_id, node_index);
        });
        assignment.len()
    }

    pub fn handle_finished(&mut self, packet: &Packet) -> bool {
        let node_id = self.get_node_id(packet, "success");
        if node_id.is_none() {
            return false;
        }
        let (key, node_id) = node_id.unwrap();
        self.scanner_task[packet.source_machine_id] = None;
        self.grid.release_grid(key);
        debug!("model_manager, scheduler, success, reset, {}, {}, {}",
                packet.source_machine_id, node_id, packet.gamma);
        true
    }

    fn get_node_id(&self, packet: &Packet, desc: &str) -> Option<(String, usize)> {
        if self.scanner_task[packet.source_machine_id].is_none() {
            debug!("model_manager, scheduler, {}, no assignment, {}, {}, {}",
                    desc, packet.packet_signature, packet.source_machine, packet.source_machine_id);
            return None;
        }
        let (key, node_id) = self.scanner_task[packet.source_machine_id].unwrap();
        if node_id != packet.node_id {
            debug!("model_manager, scheduler, {}, node_id mismatch, {}, {}, {}, {}, {}",
                    desc, packet.packet_signature, packet.source_machine, packet.source_machine_id,
                    node_id, packet.node_id);
            return None;
        }
        Some((key, node_id))
    }

    pub fn print_log(&self, num_consecutive_err: u32, gamma: &Gamma) {
        let num_working_scanners = self.scanner_task.filter(|t| t.is_some()).count();
        debug!("model_manager, scheduler, status, {}, {}, {}, {}, {}",
                num_consecutive_err, gamma.gamma, gamma.root_gamma,
                num_working_scanners, self.scanner_task.len() - num_working_scanners);
    }
}
