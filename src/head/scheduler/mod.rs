pub mod gamma;
pub mod kdtree;
pub mod packet_stats;

use commons::packet::UpdatePacket;
use commons::packet::UpdatePacketType;
use config::Config;
use head::model_with_version::ModelWithVersion;
use self::gamma::Gamma;
// use self::kdtree::Grid;
// use self::kdtree::Grids;
// use self::kdtree::KdTree;
use self::packet_stats::PacketStats;


pub struct Scheduler {
    _num_machines: usize,
    _min_grid_size: usize,

    scanner_addr: Vec<String>,
    scanner_task: Vec<Option<usize>>,  // None for idle, otherwise Some(node_id)
    _last_gamma: Vec<f32>,

    gamma: Gamma,
    packet_stats: PacketStats,

    // pub grids_version: usize,
    // curr_grids: Grids,
    // next_grids: Option<Grids>,
}


impl Scheduler {
    pub fn new(num_machines: usize, min_grid_size: usize, config: &Config) -> Scheduler {
        let gamma = Gamma::new(config.default_gamma, config.min_gamma);
        Scheduler {
            _num_machines: num_machines.clone(),
            _min_grid_size: min_grid_size,

            scanner_addr: vec![],
            scanner_task: vec![],
            _last_gamma: vec![1.0],     // ditto

            gamma: gamma,
            packet_stats: PacketStats::new(num_machines),
        }
    }

    pub fn add_scanner(&mut self, addr: String) {
        self.scanner_addr.push(addr);
        self.scanner_task.push(None);
    }

    pub fn set_assignments(
        &mut self, model: &ModelWithVersion, gamma: f32, capacity: usize,
    ) -> Vec<(String, usize)> {
        let idle_scanners: Vec<usize> =
            self.scanner_task.iter()
                .enumerate()
                .filter(|(_, assignment)| assignment.is_none())
                .map(|(scanner_index, _)| scanner_index)
                .collect();
        let num_idle_scanners = idle_scanners.len();
        let new_assigns = self.assign(idle_scanners, model, gamma, capacity);
        if num_idle_scanners > 0 {
            debug!("model-manager, assign updates, {}, {}", num_idle_scanners, new_assigns.len());
        }
        // if new_assigns.len() > 0 {
        //     let assignment: Vec<Option<usize>> = self.get_assignment();
        //     upload_assignments(&assignment, &self.exp_name);
        // }
        new_assigns
    }

    // assign a non-taken grid to each idle scanner
    fn assign(
        &mut self,
        idle_scanners: Vec<usize>,
        _model: &ModelWithVersion,
        gamma: f32,
        capacity: usize,
    ) -> Vec<(String, usize)> {
        let assignments: Vec<(usize, usize)> =
            idle_scanners.into_iter()
                         .map(|scanner_id| (scanner_id, self.get_new_grid(scanner_id, gamma)))
                         .take(capacity)
                         .collect();
        assignments.iter().for_each(|(scanner_id, node_index)| {
            self.scanner_task[*scanner_id] = Some(*node_index);
            debug!("model-manager, assign, {}, {}", scanner_id, node_index);
        });
        assignments.into_iter()
                   .map(|(scanner_id, node_index)| {
                       (self.scanner_addr[scanner_id].clone(), node_index)
                   }).collect()
    }

    pub fn handle_packet(
        &mut self,
        source_ip: &String,
        packet: &mut UpdatePacket,
        model: &mut ModelWithVersion,
        capacity: usize,
    ) -> (f32, Vec<(String, usize)>) {
        let packet_type = packet.get_packet_type();
        self.packet_stats.handle_new_packet(source_ip, &packet_type);
        match packet_type {
            UpdatePacketType::Empty => {
                self.handle_empty(packet);
            },
            UpdatePacketType::Accept => {
                self.handle_accept(packet);
            },
        }

        // refresh kdtree when gamma is too small
        self.adjust_gamma(model);
        let assigns = self.set_assignments(model, self.gamma.value(), capacity);
        // if !self.gamma.is_valid() {
        //     self.refresh_grid(self.min_grid_size);
        // }

        (self.gamma.gamma, assigns)
    }

    fn handle_accept(&mut self, packet: &UpdatePacket) -> bool {
        // self.get_grid_node_ids(packet).is_some()
        debug!("head, scheduler, empty, {}", packet.packet_id);
        true
    }

    fn handle_empty(&mut self, packet: &UpdatePacket) -> bool {
        // let grid_node_ids = self.get_grid_node_ids(packet);
        // if grid_node_ids.is_none() {
        //     return false;
        // }
        // let (grid_index, node_id) = grid_node_ids.unwrap();
        debug!("head, scheduler, empty, {}", packet.packet_id);
        // self.release_grid(grid_index);
        // callback TODO:
        // self.last_gamma[grid_index] = packet.gamma;
        true
    }

    fn adjust_gamma(&mut self, model: &mut ModelWithVersion) {
        if self.packet_stats.got_sufficient_packages() {
            if self.gamma.adjust(&self.packet_stats, model.model.size()) {
                model.update_gamma(self.gamma.gamma_version);
                // TODO: broadcast model
                // self.broadcast_model(false);
            }
            self.packet_stats.reset();
        }
    }
    
    /*
    pub fn refresh_grid(&mut self, min_size: usize) {
        let new_grid = get_new_grids(min_size, self.exp_name.as_ref());
        if new_grid.is_some() {
            self.next_grids = new_grid;
        }
        self.reset_assign();
    }

    fn reset_assign(&mut self) -> bool {
        if self.next_grids.is_some() {
            self.curr_grids = self.next_grids.take().unwrap();
            self.availability = vec![None; self.curr_grids.len()];
            self.last_gamma = vec![1.0; self.curr_grids.len()];
            self.scanner_task = vec![None; self.num_machines];
            self.grids_version += 1;
            true
        } else {
            false
        }
    }

    fn get_assignment(&self) -> Vec<Option<usize>> {
        self.scanner_task
            .iter().map(|t| {
                if t.is_some() {
                    let (_grid_index, node_index) = t.unwrap();
                    Some(node_index)
                } else {
                    None
                }
            }).collect()
    }
    */

    fn get_new_grid(&mut self, _scanner_id: usize, _gamma: f32) -> usize {
        // let mut grid_index = 0;
        // while grid_index < self.curr_grids.len() {
        //     if self.availability[grid_index].is_none() && self.last_gamma[grid_index] > gamma {
        //         break;
        //     }
        //     grid_index += 1;
        // }
        // if grid_index >= self.curr_grids.len() {
        //     return None;
        // }
        // let grid = self.curr_grids[grid_index].clone();
        // self.availability[grid_index] = Some(scanner_id);
        // Some((grid_index, grid))
        0
    }

    /*
    fn release_grid(&mut self, grid_index: usize) {
        let machine_id = self.scanner_task[grid_index];
        if machine_id.is_none() {
            return;
        }
        let machine_id = machine_id.unwrap();
        self.availability[grid_index] = None;
    }

    fn get_grid_node_ids(&self, packet: &UpdatePacket) -> Option<(usize, usize)> {
        // callback TODO:
        // if self.scanner_task[packet.source_machine_id].is_none() {
        //     debug!("model_manager, scheduler, no assignment, {}, {}, {}",
        //             packet.packet_signature, packet.source_machine, packet.source_machine_id);
        //     return None;
        // }
        // let (grid_index, node_id) = self.scanner_task[packet.source_machine_id].unwrap();
        // if node_id != packet.node_id {
        //     debug!("model_manager, scheduler, node_id mismatch, {}, {}, {}, {}, {}",
        //             packet.packet_signature, packet.source_machine, packet.source_machine_id,
        //             node_id, packet.node_id);
        //     return None;
        // }
        // Some((grid_index, node_id))
        None
    }
    */

    pub fn print_log(&self, num_consecutive_err: usize, gamma: &Gamma) {
        let num_working_scanners = self.scanner_task.iter().filter(|t| t.is_some()).count();
        debug!("model_manager, scheduler, status, {}, {}, {}, {}",
                num_consecutive_err, gamma.gamma,
                num_working_scanners, self.scanner_task.len() - num_working_scanners);
    }
}

// TODO: support loading from the local disk
/*
fn get_new_grids(min_size: usize, exp_name: &str) -> Option<Grids> {
    let ret = load_sample_s3(exp_name);
    if ret.is_none() {
        return None;
    }

    let (version, new_examples, _, _): VersionedSampleModel = ret.unwrap();
    let examples: Vec<Example> = new_examples.into_iter().map(|(e, _)| e).collect();
    debug!("scheduler, received new sample, {}, {}", version, examples.len());

    let mut kd_tree = KdTree::new(examples, min_size);
    let mut grids = kd_tree.get_leaves();
    thread_rng().shuffle(&mut grids);

    Some(grids)
}
*/


/*
#[cfg(test)]
mod tests {
    use super::Scheduler;
    use super::ModelWithVersion;

    use commons::Model;
    use commons::test_helper::get_mock_packet;

    #[test]
    fn test_scheduler() {
        let num_machines = 5;
        let test_machine_id = 0;

        let mut model = ModelWithVersion::new(Model::new(1));
        model.model.add_root(0.0, 0.0);
        let mut scheduler = Scheduler::new(num_machines, &"test".to_string(), &vec![], &mut model);

        // initial phase
        scheduler.set_assignments(&mut model, 0.5);
        let assignment = scheduler.get_assignment();
        assert_eq!(assignment.len(), num_machines);
        assert_eq!(assignment[0], Some(0));
        for i in 1..num_machines {
            assert_eq!(assignment[i], None);
        }

        let packet = get_mock_packet(test_machine_id, 0, 0.5, 0);
        scheduler.handle_accept(&packet);
        assert_eq!(scheduler.get_assignment()[test_machine_id], Some(0));
        scheduler.handle_empty(&packet);
        assert_eq!(scheduler.get_assignment()[test_machine_id], None);
        scheduler.set_assignments(&mut model, 0.5);
        assert_eq!(scheduler.get_assignment()[test_machine_id], None);  // because \gamma didn't change
        scheduler.set_assignments(&mut model, 0.4);
        assert_eq!(scheduler.get_assignment()[test_machine_id], Some(0));

        // refresh grid
        scheduler.refresh_grid(10);
        scheduler.set_assignments(&mut model, 0.5);
        let assignment = scheduler.get_assignment();
        assert_eq!(assignment.len(), num_machines);
        for i in 0..num_machines {
            assert!(assignment[i].is_some());
        }
        let mut assigns: Vec<usize> = assignment.iter().map(|t| t.unwrap()).collect();
        let assign0 = assigns[test_machine_id];
        // all assignments are unique
        assigns.sort();
        for i in 1..num_machines {
            assert!(assigns[i] != assigns[i - 1])
        }

        let packet = get_mock_packet(test_machine_id, assign0, 0.5, 0);
        scheduler.handle_accept(&packet);
        assert_eq!(scheduler.get_assignment()[test_machine_id], Some(assign0));
        scheduler.handle_empty(&packet);
        let assignment = scheduler.get_assignment();
        assert_eq!(assignment[test_machine_id], None);
        for i in (test_machine_id + 1)..num_machines {
            assert!(assignment[i].is_some());
        }
        // now we have enough grids, so no need to set lower gamma (yet)
        scheduler.set_assignments(&model, 0.5);
        assert!(scheduler.get_assignment()[test_machine_id].is_some());
    }
}
*/

// let curr_acc_rate = model_sync.packet_stats.as_ref().unwrap().avg_accept_rate;
// let new_acc_rate = model_sync.packet_stats.as_ref().unwrap().avg_accept_rate;
// assert!(new_acc_rate < curr_acc_rate);