pub mod packet_stats;

use std::sync::mpsc;

// #[cfg(not(test))] use tmsn::network::start_network_only_recv;
// #[cfg(test)]
use commons::test_helper::start_network_only_recv;

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
    model_ts: f32,
    num_trees: usize,
    exp_name: String,
    min_ess: f32,
    min_grid_size: usize,

    gamma: Gamma,
    next_model_sender: Sender<(Model, String)>,
    scheduler: Scheduler,
    packet_receiver: Option<mpsc::Receiver<Packet>>,
    packet_stats: Option<PacketStats>,

    _performance_mon: PerformanceMonitor,
    _last_logging_ts: f32,
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
        bins: &Vec<Bins>,
        num_scanners: usize,
    ) -> ModelSync {
        let mut model = ModelWithVersion::new(init_tree.clone());
        let scheduler = Scheduler::new(num_scanners, exp_name, bins, &mut model);
        ModelSync {
            model: model,
            model_ts: 0.0,

            // Configurations
            num_trees: num_trees,
            exp_name: exp_name.clone(),
            min_ess: min_ess,
            min_grid_size: min_grid_size,

            // cluster status
            gamma: gamma,

            // Shared variables
            next_model_sender: next_model_sender,

            scheduler: scheduler,
            packet_stats: None,
            packet_receiver: None,

            _performance_mon: PerformanceMonitor::new(),
            _last_logging_ts: 0.0,
        }
    }

    pub fn run_with_network(&mut self, machine_name: String, remote_ips: Vec<String>, port: u16) {
        self.start_network(machine_name, remote_ips, port);
        let mut num_consecutive_err = 0;
        while self.continue_training() {
            self.sync_once(&mut num_consecutive_err);
            self.print_log(num_consecutive_err);
        }

        info!("Model sync quits, {}, Model length, {}, Is gamma significant, {}",
                self.continue_training(), self.model.model.size(), self.gamma.is_valid());
        let final_model = write_model(&self.model.model, self.model_ts, false);
        debug!("model_manager, final model, {}", final_model);

        // send quit signal to the sampler which runs on the same machine
        {
            let filename = "status.txt".to_string();
            write_all(&filename, "0".as_bytes()).unwrap();
        }
    }

    fn start_network(
        &mut self, machine_name: String, remote_ips: Vec<String>, port: u16,
    ) -> mpsc::Sender<Packet> {
        let (packet_stats, packet_sender, packet_receiver) =
            start_network(machine_name, remote_ips, port);
        self.packet_stats = Some(packet_stats);
        self.packet_receiver = Some(packet_receiver);
        self._performance_mon.start();
        self.broadcast_model(true);
        packet_sender
    }

    fn sync_once(&mut self, num_consecutive_err: &mut usize) {
        // Handle packets
        let packet = self.packet_receiver.as_mut().unwrap().try_recv();
        if packet.is_err() {
            *num_consecutive_err += 1;
        } else {
            self.handle_packet(&mut packet.unwrap());
            *num_consecutive_err = 0;
        }
        // refresh kdtree when gamma is too small
        self.adjust_gamma();
        self.scheduler.set_assignments(&mut self.model, self.gamma.value());
        if !self.gamma.is_valid() {
            self.scheduler.refresh_grid(self.min_grid_size);
        }
    }


    fn continue_training(&self) -> bool {
        // TODO: model.tree_size should not consider the tree nodes added by kd-tree
        self.gamma.is_valid() && (
            self.num_trees <= 0 || self.model.model.tree_size < self.num_trees)
    }


    fn handle_packet(&mut self, packet: &mut Packet) {
        let mut packet_stats = self.packet_stats.take().unwrap();
        packet.source_machine_id =
            packet.source_machine_id % packet_stats.num_machines;
        let packet_type = packet.get_packet_type(self.min_ess);
        packet_stats.handle_new_packet(packet, &packet_type);
        match packet_type {
            PacketType::SmallEffSize => {
                // Ignore updates generated on a small-ess sample
            },
            PacketType::Empty => {
                self.scheduler.handle_empty(packet);
            },
            PacketType::Accept => {
                self.model_ts = self._performance_mon.get_duration();
                self.update_model(packet);
                self.scheduler.handle_accept(packet);
            },
        }
        self.packet_stats = Some(packet_stats);
    }


    fn adjust_gamma(&mut self) {
        let mut packet_stats = self.packet_stats.take().unwrap();
        if packet_stats.got_sufficient_packages() {
            if self.gamma.adjust(&packet_stats, self.model.model.size()) {
                self.model.update_gamma(self.gamma.gamma_version);
                self.broadcast_model(false);
            }
            packet_stats.reset();
        }
        self.packet_stats = Some(packet_stats);
    }


    fn broadcast_model(&mut self, is_model_updated: bool) {
        let is_upload_success = upload_model(
            &self.model.model, &self.model.model_sig, self.gamma.gamma, &self.exp_name);
        if is_model_updated {
            self.next_model_sender.send(
                (self.model.model.clone(), self.model.model_sig.clone()));
            write_model(&self.model.model, self.model_ts, true);
        }
        debug!("model_manager, upload model, {}, {}",
                is_upload_success, self.model.model_sig);
    }


    fn update_model(&mut self, packet: &Packet) -> Vec<usize> {
        assert!(packet.updates.size > 0);
        let (new_node_indices, count_new, count_updates) = self.model.update(
            &packet.updates, &packet.this_model_signature, self.gamma.gamma);
        self.broadcast_model(true);
        debug!("model_manager, new updates, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                self.model.model.tree_size, self.model.model.size(),
                packet.packet_signature, packet.source_machine_id, packet.node_id,
                self.model.model.depth[packet.node_id], packet.gamma, packet.updates.size,
                count_new, count_updates);
        new_node_indices
    }


    fn print_log(&mut self, num_consecutive_err: usize) {
        if self._performance_mon.get_duration() - self._last_logging_ts >= 10.0 {
            self.scheduler.print_log(num_consecutive_err, &self.gamma);
            self.packet_stats.as_ref().unwrap().print_log();
            self.model.print_log();
            self._last_logging_ts = self._performance_mon.get_duration();
        }
    }
}


// start a network _receiver_ on the master node, which receives packages from the scanners
fn start_network(
    machine_name: String, remote_ips: Vec<String>, port: u16,
) -> (PacketStats, mpsc::Sender<Packet>, mpsc::Receiver<Packet>) {
    let (packet_s, packet_r): (mpsc::Sender<Packet>, mpsc::Receiver<Packet>) = mpsc::channel();
    start_network_only_recv(machine_name.as_ref(), &remote_ips, port, packet_s.clone()).unwrap();
    (PacketStats::new(remote_ips.len()), packet_s, packet_r)
}


#[cfg(test)]
mod tests {
    use super::ModelSync;
    use commons::Model;
    use commons::channel::Receiver;
    use master::model_manager::gamma::Gamma;

    use commons::channel;
    use commons::test_helper::get_mock_packet;


    #[test]
    fn test_model_sync() {
        let mut model = Model::new(1);
        model.add_root(0.0, 0.0);
        let gamma = Gamma::new(0.5, 0.05);
        let (next_model_s, mut next_model_r) = channel::bounded(100, "updated-models");
        let mut model_sync = ModelSync::new(
            &model,
            100,
            &"test".to_string(),
            0.1,
            10,
            gamma,
            next_model_s,
            &vec![],
            5,
        );
        let packet_sender = model_sync.start_network(
            "tester".to_string(), vec!["s1".to_string(), "s2".to_string()], 8000);
        assert!(try_recv(&mut next_model_r).is_some());

        let mut num_consecutive_err = 0;
        // no packet sent
        model_sync.sync_once(&mut num_consecutive_err);
        model_sync.sync_once(&mut num_consecutive_err);
        assert_eq!(num_consecutive_err, 2);

        // send a good packet
        let packet = get_mock_packet(0, 0, 0.5, 1);
        assert!(try_recv(&mut next_model_r).is_none());
        packet_sender.send(packet).unwrap();
        model_sync.sync_once(&mut num_consecutive_err);
        assert_eq!(num_consecutive_err, 0);
        assert!(try_recv(&mut next_model_r).is_some());

        // send many empty packets
        assert!(try_recv(&mut next_model_r).is_none());
        let curr_acc_rate = model_sync.packet_stats.as_ref().unwrap().avg_accept_rate;
        for _ in 0..10 {
            let packet = get_mock_packet(0, 0, 0.5, 0);
            packet_sender.send(packet).unwrap();
            model_sync.sync_once(&mut num_consecutive_err);
        }
        let new_acc_rate = model_sync.packet_stats.as_ref().unwrap().avg_accept_rate;
        assert_eq!(num_consecutive_err, 0);
        assert!(try_recv(&mut next_model_r).is_none());
        assert!(new_acc_rate < curr_acc_rate);
    }

    fn try_recv<T>(receiver: &mut Receiver<T>) -> Option<T> {
        for _ in 0..3 {
            let item = receiver.try_recv();
            if item.is_some() {
                return item;
            }
        }
        None
    }
}
