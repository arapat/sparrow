pub mod kdtree;

use rand::Rng;
use rand::thread_rng;

use commons::bins::Bins;
use commons::packet::Packet;
use commons::persistent_io::VersionedSampleModel;
use commons::persistent_io::load_sample_s3;
use commons::persistent_io::upload_assignments;
use Example;
use super::gamma::Gamma;
use super::model_with_version::ModelWithVersion;
use self::kdtree::Grid;
use self::kdtree::Grids;
use self::kdtree::KdTree;

pub struct Scheduler {
    exp_name: String,
    num_machines: usize,

    scanner_task: Vec<Option<(usize, usize)>>,  // (key, node_id)
    availability: Vec<Option<usize>>,
    next_available: usize,

    pub grids_version: usize,
    curr_grids: Grids,
    next_grids: Option<Grids>,
}


impl Scheduler {
    pub fn new(
        num_machines: usize, exp_name: &String, _bins: &Vec<Bins>, model: &mut ModelWithVersion,
    ) -> Scheduler {
        let mut scheduler = Scheduler {
            exp_name: exp_name.clone(),
            num_machines: num_machines.clone(),
            scanner_task: vec![None; num_machines],
            grids_version: 0,
            curr_grids: vec![],
            availability: vec![],
            next_available: 0,
            next_grids: None,
        };
        scheduler.update(model);
        scheduler
    }

    pub fn update(&mut self, model: &mut ModelWithVersion) -> usize {
        let idle_scanners: Vec<usize> =
            self.scanner_task.iter()
                .enumerate()
                .filter(|(_, assignment)| assignment.is_none())
                .map(|(scanner_index, _)| scanner_index)
                .collect();
        let num_updates = self.assign(idle_scanners, model);
        if num_updates > 0 {
            debug!("model-manager, assign updates, {}", num_updates);
            let assignment: Vec<Option<usize>> =
                self.scanner_task
                    .iter().map(|t| {
                        if t.is_some() {
                            Some(t.unwrap().1)
                        } else {
                            None
                        }
                    }).collect();
            upload_assignments(&assignment, &self.exp_name);
        }
        num_updates
    }

    pub fn handle_accept(&mut self, packet: &Packet) -> bool {
        self.handle_packet(packet, "success")
    }

    pub fn handle_empty(&mut self, packet: &Packet) -> bool {
        self.handle_packet(packet, "empty")
    }

    pub fn refresh_grid(&mut self, min_size: usize) {
        let new_grid = get_new_grids(min_size, self.exp_name.as_ref());
        if new_grid.is_some() {
            self.next_grids = new_grid;
        }
        self.reset_assign();
    }

    fn handle_packet(&mut self, packet: &Packet, msg: &str) -> bool {
        let node_id = self.get_node_id(packet, "success");
        if node_id.is_none() {
            return false;
        }
        let (grid_index, node_id) = node_id.unwrap();
        self.release_grid(grid_index);
        debug!("model_manager, scheduler, {}, reset, {}, {}, {}",
                msg, packet.source_machine_id, node_id, packet.gamma);
        true
    }

    fn reset_assign(&mut self) -> bool {
        if self.next_grids.is_some() {
            self.curr_grids = self.next_grids.take().unwrap();
            self.availability = vec![None; self.curr_grids.len()];
            self.scanner_task = vec![None; self.num_machines];
            self.next_available = 0;
            self.grids_version += 1;
            true
        } else {
            false
        }
    }

    // assign a non-taken grid to each idle scanner
    fn assign(&mut self, idle_scanners: Vec<usize>, model: &mut ModelWithVersion) -> usize {
        let assignment: Vec<(usize, (usize, Grid))> =
            idle_scanners.into_iter()
                         .map(|scanner_id| (scanner_id, self.get_new_grid(scanner_id)))
                         .collect();
        let update_size = assignment.len();
        assignment.into_iter().for_each(|(scanner_id, (grid_index, grid))| {
            let node_index = model.add_grid(grid);
            self.scanner_task[scanner_id] = Some((grid_index, node_index));
            debug!("model-manager, assign, {}, {}, {}", scanner_id, grid_index, node_index);
        });
        update_size
    }

    fn get_new_grid(&mut self, scanner_id: usize) -> (usize, Grid) {
        let grid_index = self.next_available;
        let grid = self.curr_grids[grid_index].clone();
        self.availability[self.next_available] = Some(scanner_id);
        self.next_available += 1;
        (grid_index, grid)
    }

    fn release_grid(&mut self, grid_index: usize) {
        let machine_id = self.availability[grid_index];
        if machine_id.is_none() {
            return;
        }
        let machine_id = machine_id.unwrap();
        self.availability[grid_index] = None;
        self.scanner_task[machine_id] = None;
    }

    fn get_node_id(&self, packet: &Packet, desc: &str) -> Option<(usize, usize)> {
        if self.scanner_task[packet.source_machine_id].is_none() {
            debug!("model_manager, scheduler, {}, no assignment, {}, {}, {}",
                    desc, packet.packet_signature, packet.source_machine, packet.source_machine_id);
            return None;
        }
        let (grid_index, node_id) = self.scanner_task[packet.source_machine_id].unwrap();
        if node_id != packet.node_id {
            debug!("model_manager, scheduler, {}, node_id mismatch, {}, {}, {}, {}, {}",
                    desc, packet.packet_signature, packet.source_machine, packet.source_machine_id,
                    node_id, packet.node_id);
            return None;
        }
        Some((grid_index, node_id))
    }

    pub fn print_log(&self, num_consecutive_err: u32, gamma: &Gamma) {
        let num_working_scanners = self.scanner_task.iter().filter(|t| t.is_some()).count();
        debug!("model_manager, scheduler, status, {}, {}, {}, {}",
                num_consecutive_err, gamma.gamma,
                num_working_scanners, self.scanner_task.len() - num_working_scanners);
    }
}

// TODO: support loading from the local disk
fn get_new_grids(min_size: usize, exp_name: &str) -> Option<Grids> {
    let ret = load_sample_s3(0, exp_name);
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
