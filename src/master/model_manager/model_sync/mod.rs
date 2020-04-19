pub mod packet_stats;

use std::sync::mpsc;

use tmsn::network::start_network_only_recv;

use commons::Model;
use commons::bins::Bins;
use commons::channel::Sender;
use commons::packet::Packet;
use commons::packet::PacketType;
use commons::performance_monitor::PerformanceMonitor;

use commons::io::write_all;
use commons::persistent_io::upload_model;
use commons::persistent_io::write_model;

use super::gamma::Gamma;
use super::model_with_version::ModelWithVersion;
use super::scheduler::Scheduler;

use self::packet_stats::PacketStats;


pub struct ModelSync {
    model: ModelWithVersion,
    num_trees: usize,
    exp_name: String,
    min_ess: f32,
    min_grid_size: usize,

    gamma: Gamma,
    next_model_sender: Sender<(Model, String)>,

    packet_stats: Option<PacketStats>,
    packet_receiver: Option<mpsc::Receiver<Packet>>,
}


impl ModelSync {
    pub fn new(
        init_tree: &Model,
        num_trees: usize,
        exp_name: &String,
        min_ess: f32,
        min_grid_size: usize,
        gamma: Gamma,
        next_model_sender: Sender<(Model, String)>,
    ) -> ModelSync {
        let model = ModelWithVersion::new(init_tree.clone(), num_trees);
        ModelSync {
            model: model,

            // Configurations
            num_trees: num_trees,
            exp_name: exp_name.clone(),
            min_ess: min_ess,
            min_grid_size: min_grid_size,

            // cluster status
            gamma: gamma,

            // Shared variables
            next_model_sender: next_model_sender,

            packet_receiver: None,
            packet_stats: None,
        }
    }


    pub fn start_network(&mut self, machine_name: String, remote_ips: Vec<String>, port: u16) {
        self.broadcast_model(0.0, true);
        let (packet_s, packet_r): (mpsc::Sender<Packet>, mpsc::Receiver<Packet>) =
            mpsc::channel();
        start_network_only_recv(machine_name.as_ref(), &remote_ips, port, packet_s).unwrap();
        self.packet_receiver = Some(packet_r);
        self.packet_stats = Some(PacketStats::new(remote_ips.len()));
    }


    pub fn run_with_network(&mut self, bins: Vec<Bins>) {
        let global_timer = PerformanceMonitor::new();
        let mut _last_cluster_update = global_timer.get_duration();
        let mut packet_stats = self.packet_stats.take().unwrap();
        let mut scheduler = Scheduler::new(
            packet_stats.num_machines, &self.exp_name, &bins, &mut self.model);
        let packet_receiver = self.packet_receiver.take().unwrap();

        // start listening to network
        let mut global_timer = PerformanceMonitor::new();
        global_timer.start();
        let mut num_consecutive_err = 0;
        let mut last_model_timestamp = global_timer.get_duration();
        let mut last_logging_timestamp = global_timer.get_duration();
        while self.continue_training() {
            if global_timer.get_duration() - last_logging_timestamp >= 10.0 {
                scheduler.print_log(num_consecutive_err, &self.gamma);
                packet_stats.print_log();
                self.model.print_log();
                last_logging_timestamp = global_timer.get_duration();
            }

            // adjust gamma
            if packet_stats.is_triggered() {
                if self.gamma.adjust(&packet_stats, self.model.model.size()) {
                    self.model.update_gamma(self.gamma.gamma_version);
                    self.broadcast_model(last_model_timestamp, false);
                    // TODO: should we allow re-assessing all tree nodes if we have increased gamma,
                    // by setting `last_failed_gamma` in `node_status` to 1.0
                }
                packet_stats.reset();
            }

            // Update assignments
            let _num_updates = scheduler.update(&mut self.model);

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
            let packet_type = packet.get_packet_type(self.min_ess);
            packet_stats.handle_new_packet(&packet, &packet_type);
            match packet_type {
                PacketType::SmallEffSize => {
                    // Ignore updates generated on a small-ess sample
                },
                PacketType::Empty => {
                    self.gamma.decrease_gamma(self.model.model.size());
                    scheduler.handle_empty(&packet);
                },
                PacketType::Accept => {
                    last_model_timestamp = global_timer.get_duration();
                    self.update_model(&packet, last_model_timestamp);
                    scheduler.handle_accept(&packet);
                },
            }

            // refresh kdtree when gamma is too small
            if !self.gamma.is_valid() {
                scheduler.refresh_grid(self.min_grid_size);
            }
        }

        info!("Model sync quits, {}, Model length, {}, Is gamma significant, {}",
                self.continue_training(), self.model.model.size(), self.gamma.is_valid());
        let final_model = write_model(&self.model.model, last_model_timestamp, false);
        debug!("model_manager, final model, {}", final_model);

        // send quit signal to the sampler which runs on the same machine
        {
            let filename = "status.txt".to_string();
            write_all(&filename, "0".as_bytes()).unwrap();
        }
    }


    fn broadcast_model(&mut self, last_timestamp: f32, is_model_updated: bool) {
        let is_upload_success = upload_model(
            &self.model.model, &self.model.model_sig, self.gamma.gamma, &self.exp_name);
        if is_model_updated {
            self.next_model_sender.send(
                (self.model.model.clone(), self.model.model_sig.clone()));
            write_model(&self.model.model, last_timestamp, true);
        }
        debug!("model_manager, upload model, {}, {}",
                is_upload_success, self.model.model_sig);
    }


    fn update_model(
        &mut self, packet: &Packet, last_timestamp: f32,
    ) -> Vec<usize> {
        assert!(packet.updates.size > 0);
        let (new_node_indices, count_new, count_updates) = self.model.update(
            &packet.updates, &packet.this_model_signature, self.gamma.gamma);
        self.broadcast_model(last_timestamp, true);
        debug!("model_manager, new updates, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                self.model.model.tree_size, self.model.model.size(),
                packet.packet_signature, packet.source_machine_id, packet.node_id,
                self.model.model.depth[packet.node_id], packet.gamma, packet.updates.size,
                count_new, count_updates);
        new_node_indices
    }


    fn continue_training(&self) -> bool {
        self.gamma.is_valid() && (
            self.num_trees <= 0 || self.model.model.tree_size < self.num_trees)
    }
}