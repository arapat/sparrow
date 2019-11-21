pub mod gamma;
pub mod packet_stats;
mod scheduler;

use std::sync::Arc;
use std::sync::RwLock;
use std::sync::mpsc;

use tmsn::network::start_network_only_recv;

use commons::INIT_MODEL_PREFIX;
use commons::Model;
use commons::channel::Sender;
use commons::packet::Packet;
use commons::packet::PacketType;
use commons::performance_monitor::PerformanceMonitor;
use commons::tree::UpdateList;

use commons::persistent_io::upload_model;
use commons::persistent_io::write_model;

use self::gamma::Gamma;
use self::packet_stats::PacketStats;
use self::scheduler::Scheduler;


pub struct ModelStats {
    pub model: Model,
    model_prefix: String,
    gamma_version: usize,
    pub model_sig: String,
    pub avail_nodes: usize,
    pub avail_new_tree: usize,

    pub max_num_trees: usize,
    pub max_depth: usize,
}


impl ModelStats {
    pub fn new(model: Model, max_num_trees: usize, max_depth: usize) -> ModelStats {
        let mut avail_nodes = 0;
        let mut avail_new_tree = 0;
        if model.tree_size > 0 && max_depth > 0 && model.children[0].len() < max_num_trees {
            avail_nodes += 1;
        }
        for i in 1..model.tree_size {
            let (depth, num_child) = (model.depth[i], model.children[i].len());
            if num_child <= 0 && depth == 1 {
                avail_new_tree += 1;
            }
            if num_child <= 0 && depth < max_depth {
                avail_nodes += 1;
            }
        }

        let (model_prefix, gamma_version) = (INIT_MODEL_PREFIX.to_string(), 0);
        let model_sig = get_model_sig(&model_prefix, gamma_version);
        ModelStats {
            model: model,
            model_prefix: model_prefix,
            gamma_version: gamma_version,
            model_sig: model_sig,
            avail_nodes: avail_nodes,
            avail_new_tree: avail_new_tree,

            max_num_trees: max_num_trees,
            max_depth: max_depth,
        }
    }

    fn update(
        &mut self, patch: &UpdateList, new_prefix: &String, gamma: f32,
    ) -> (Vec<usize>, usize, usize) {
        let node_indices: Vec<(usize, bool)> = self.model.append_patch(&patch, gamma, true);
        let new_nodes_depth: Vec<usize> = node_indices.iter()
                                                      .map(|(i, _)| self.model.depth[*i])
                                                      .collect();
        let (count_new, count_updates) = patch.is_new.iter().fold(
            (0, 0), |(new, old), t| { if *t { (new + 1, old) } else { (new, old + 1) } });
        self.avail_nodes    += new_nodes_depth.len();
        self.avail_new_tree += new_nodes_depth.iter().filter(|depth| **depth == 1).count();
        self.model_prefix = new_prefix.clone();
        self.model_sig = get_model_sig(&self.model_prefix, self.gamma_version);
        (node_indices.into_iter().map(|(i, _)| i).collect(), count_new, count_updates)
    }

    fn update_gamma(&mut self, gamma_version: usize) {
        self.gamma_version = gamma_version;
        self.model_sig = get_model_sig(&self.model_prefix, self.gamma_version);
    }

    fn print_log(&self) {
        let num_roots = self.model.depth.iter().filter(|t| **t == 1).count();
        debug!("model stats, status, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                self.model.tree_size,
                self.model.size(),
                num_roots,
                self.model_prefix,
                self.gamma_version,
                self.model_sig,
                self.avail_nodes,
                self.avail_new_tree,
                self.max_num_trees,
                self.max_depth);
    }
}


fn get_model_sig(prefix: &String, gamma_version: usize) -> String {
    format!("{}_{}", prefix, gamma_version)
}


pub struct ModelSync {
    model_stats: ModelStats,
    num_iterations: usize,
    exp_name: String,

    gamma: Gamma,
    sampler_state: Arc<RwLock<bool>>,
    next_model_sender: Sender<(Model, String)>,
    current_sample_version: Arc<RwLock<usize>>,
    node_counts: Arc<RwLock<Vec<u32>>>,

    packet_stats: Option<PacketStats>,
    packet_receiver: Option<mpsc::Receiver<Packet>>,
}


impl ModelSync {
    pub fn new(
        model_stats: ModelStats,
        num_iterations: usize,
        exp_name: &String,
        gamma: Gamma,
        sampler_state: Arc<RwLock<bool>>,
        next_model_sender: Sender<(Model, String)>,
        current_sample_version: Arc<RwLock<usize>>,
        node_counts: Arc<RwLock<Vec<u32>>>,
    ) -> ModelSync {

        ModelSync {
            model_stats: model_stats,

            // Configurations
            num_iterations: num_iterations,
            exp_name: exp_name.clone(),

            // cluster status
            gamma: gamma,

            // Shared variables
            sampler_state: sampler_state,
            next_model_sender: next_model_sender,
            current_sample_version: current_sample_version,
            node_counts: node_counts,

            packet_receiver: None,
            packet_stats: None,
        }

        // TODO: remove or resume the trackers
        // let mut node_sum_gamma_sq = vec![0.0; model.tree_size];
        // let mut node_timestamp = vec![0.0; model.tree_size];
    }


    pub fn start_network(&mut self, machine_name: String, remote_ips: Vec<String>, port: u16) {
        self.broadcast_model(0.0, true);
        let (packet_s, packet_r): (mpsc::Sender<Packet>, mpsc::Receiver<Packet>) =
            mpsc::channel();
        start_network_only_recv(machine_name.as_ref(), &remote_ips, port, packet_s);
        self.packet_receiver = Some(packet_r);
        self.packet_stats = Some(PacketStats::new(remote_ips.len()));
    }


    pub fn run_with_network(&mut self) {
        let global_timer = PerformanceMonitor::new();
        let mut _last_cluster_update = global_timer.get_duration();
        let mut packet_stats = self.packet_stats.take().unwrap();
        let mut scheduler = Scheduler::new(
            packet_stats.num_machines, &self.model_stats, &self.exp_name, &self.gamma);
        let packet_receiver = self.packet_receiver.take().unwrap();

        // start listening to network
        let mut global_timer = PerformanceMonitor::new();
        global_timer.start();
        let mut num_consecutive_err = 0;
        let mut last_model_timestamp = global_timer.get_duration();
        let mut last_logging_timestamp = global_timer.get_duration();
        while self.continue_training() {
            if global_timer.get_duration() - last_logging_timestamp >= 10.0 {
                scheduler.print_log(num_consecutive_err, &self.model_stats, &self.gamma);
                packet_stats.print_log();
                self.model_stats.print_log();
                last_logging_timestamp = global_timer.get_duration();
            }

            // adjust gamma
            if packet_stats.is_triggered(self.model_stats.avail_nodes) {
                if self.gamma.adjust(&packet_stats, self.model_stats.model.size()) {
                    self.model_stats.update_gamma(self.gamma.gamma_version);
                    self.broadcast_model(last_model_timestamp, false);
                    // TODO: should we allow re-assessing all tree nodes if we have increased gamma,
                    // by setting `last_failed_gamma` in `node_status` to 1.0
                }
                packet_stats.reset();
            }

            // Update assignments
            let (_num_updates, force_dec_gamma) = scheduler.update(&self.model_stats, &self.gamma);
            if force_dec_gamma {
                self.gamma.decrease_gamma(self.model_stats.model.size());
            }

            // Handle packets
            let packet = packet_receiver.try_recv();
            if packet.is_err() {
                // debug!("model_manager, packet error, {:?}", packet);
                num_consecutive_err += 1;
                continue;
            }
            num_consecutive_err = 0;
            let mut packet: Packet = packet.unwrap();
            packet.source_machine_id = packet.source_machine_id % packet_stats.num_machines;
            let packet_type = packet.get_packet_type(
                &self.current_sample_version, &self.model_stats.model_sig);
            packet_stats.handle_new_packet(&packet, &packet_type);
            let node_count = self.get_node_counts(packet.node_id);
            match packet_type {
                PacketType::EmptyRoot | PacketType::EmptyNonroot => {
                    scheduler.handle_failure(&packet, &self.gamma, node_count);
                    if packet.node_id == 0 {  // is root
                        self.gamma.decrease_root_gamma(self.model_stats.model.size());
                        self.model_stats.update_gamma(self.gamma.gamma_version);
                        self.broadcast_model(last_model_timestamp, false);
                    }
                },
                PacketType::AcceptNonroot | PacketType::AcceptRoot => {
                    last_model_timestamp = global_timer.get_duration();
                    let new_node_indices =
                        self.update_model(&packet, node_count, last_model_timestamp);
                    scheduler.append_new_nodes(&new_node_indices);
                    if scheduler.handle_success(
                        &packet, &self.model_stats, &self.gamma, node_count) {
                        // the tree node can no longer be extended
                        self.model_stats.avail_nodes -= 1;
                        if self.model_stats.model.depth[packet.node_id] == 1 {  // is root
                            self.model_stats.avail_new_tree -= 1;
                        }
                    }
                },
                PacketType::RejectBaseModel | PacketType::RejectSample => {
                    // TODO: Reject because of the sample version might not be necessary
                },
            }
        }

        info!("Model sync quits, {}, Model length, {}, Is gamma significant, {}",
                self.continue_training(), self.model_stats.model.size(), self.gamma.is_valid());
        let final_model = write_model(&self.model_stats.model, last_model_timestamp, false);
        debug!("model_manager, final model, {}", final_model);
        {
            debug!("sampler state, false, model sync quits");
            let mut state = self.sampler_state.write().unwrap();
            *state = false;
        }
    }


    fn broadcast_model(&mut self, last_timestamp: f32, is_model_updated: bool) {
        let is_upload_success = upload_model(
            &self.model_stats.model, &self.model_stats.model_sig,
            self.gamma.gamma, self.gamma.root_gamma,
            &self.exp_name);
        if is_model_updated {
            self.next_model_sender.send(
                (self.model_stats.model.clone(), self.model_stats.model_sig.clone()));
            write_model(&self.model_stats.model, last_timestamp, true);
        }
        debug!("model_manager, upload model, {}, {}",
                is_upload_success, self.model_stats.model_sig);
    }


    fn update_model(
        &mut self, packet: &Packet, node_count: u32, last_timestamp: f32,
    ) -> Vec<usize> {
        assert!(packet.updates.size > 0);
        let (new_node_indices, count_new, count_updates) = self.model_stats.update(
            &packet.updates, &packet.this_model_signature, self.gamma.gamma);
        self.broadcast_model(last_timestamp, true);
        debug!("model_manager, new updates, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                self.model_stats.model.tree_size, self.model_stats.model.size(),
                packet.packet_signature, packet.source_machine_id, packet.node_id, node_count,
                self.model_stats.model.depth[packet.node_id], packet.gamma, packet.updates.size,
                count_new, count_updates);
        new_node_indices
    }


    fn get_node_counts(&self, node_id: usize) -> u32 {
        let c = self.node_counts.read().unwrap();
        if node_id >= c.len() {
            0
        } else {
            c[node_id]
        }
    }

    fn continue_training(&self) -> bool {
        let t = self.sampler_state.read().unwrap();
        *t && self.gamma.is_valid() && (
            self.num_iterations <= 0 || self.model_stats.model.size() < self.num_iterations)
    }
}