pub mod model_with_version;

use std::sync::mpsc;

// TODO: replace network
/* #[cfg(not(test))] use tmsn::network::start_network_only_recv;
// #[cfg(test)] */ use commons::test_helper::start_network_only_recv;

use commons::Model;
use commons::bins::Bins;
use commons::channel::Sender;
use commons::packet::UpdatePacket;
use commons::packet::UpdatePacketType;
use commons::performance_monitor::PerformanceMonitor;

use commons::io::write_all;
use commons::persistent_io::write_model;

use self::model_with_version::ModelWithVersion;


pub struct ModelSync {
    model: ModelWithVersion,
    model_ts: f32,
    num_trees: usize,
    exp_name: String,
    min_ess: f32,

    next_model_sender: Sender<(Model, String)>,
    packet_receiver: Option<mpsc::Receiver<(String, UpdatePacket)>>,

    _performance_mon: PerformanceMonitor,
    _last_logging_ts: f32,
}


impl ModelSync {
    pub fn new(
        init_tree: &Model,
        num_trees: usize,
        exp_name: &String,
        min_ess: f32,
        next_model_sender: Sender<(Model, String)>,
        bins: &Vec<Bins>,
        num_scanners: usize,
    ) -> ModelSync {
        let mut model = ModelWithVersion::new(init_tree.clone(), "Sampler".to_string());
        ModelSync {
            model: model,
            model_ts: 0.0,

            // Configurations
            num_trees: num_trees,
            exp_name: exp_name.clone(),
            min_ess: min_ess,

            // Shared variables
            next_model_sender: next_model_sender,

            packet_receiver: None,

            _performance_mon: PerformanceMonitor::new(),
            _last_logging_ts: 0.0,
        }
    }

    fn handle_packet(&mut self, source_ip: &String, packet: &mut UpdatePacket) {
        if packet.get_packet_type() == UpdatePacketType::Accept {
            self.model_ts = self._performance_mon.get_duration();
            self.update_model(&source_ip, &packet);
        }
    }


    fn continue_training(&self) -> bool {
        // TODO: model.tree_size should not consider the tree nodes added by kd-tree
        self.num_trees <= 0 || self.model.model.tree_size < self.num_trees
    }


    fn broadcast_model(&mut self, is_model_updated: bool) {
        // callback TODO: fix upload model
        // let is_upload_success = upload_model(
        //     &self.model.model, &self.model.model_sig, self.gamma.gamma, &self.exp_name);
        if is_model_updated {
            // callback TODO: next_model_sender
            // self.next_model_sender.send(
            //     (self.model.model.clone(), self.model.model_sig.clone()));
            write_model(&self.model.model, self.model_ts, true);
        }
        // debug!("model_manager, upload model, {}, {}",
        //         is_upload_success, self.model.model_sig);
    }


    fn update_model(&mut self, last_update_from: &String, packet: &UpdatePacket) -> Vec<usize> {
        assert!(packet.updates.size > 0);
        let (new_node_indices, count_new, count_updates) =
            self.model.update(&packet.updates, last_update_from);
        self.broadcast_model(true);
        debug!("model_manager, new updates, {}, {}, {}, {}, {}",
                self.model.model.tree_size,
                self.model.model.size(),
                packet.updates.size,
                count_new, count_updates);
        new_node_indices
    }


    fn print_log(&mut self) {
        if self._performance_mon.get_duration() - self._last_logging_ts >= 10.0 {
            self.model.print_log();
            self._last_logging_ts = self._performance_mon.get_duration();
        }
    }
}


// start a network _receiver_ on the master node, which receives packages from the scanners
fn start_network(
    machine_name: String, remote_ips: Vec<String>, port: u16,
) -> (mpsc::Sender<(String, UpdatePacket)>, mpsc::Receiver<(String, UpdatePacket)>) {
    let (packet_s, packet_r):
        (mpsc::Sender<(String, UpdatePacket)>, mpsc::Receiver<(String, UpdatePacket)>) =
        mpsc::channel();
    start_network_only_recv(machine_name.as_ref(), &remote_ips, port, packet_s.clone()).unwrap();
    (packet_s, packet_r)
}


#[cfg(test)]
mod tests {
    use super::ModelSync;
    use commons::Model;
    use commons::channel::Receiver;

    use commons::channel;
    use commons::test_helper::get_mock_packet;


    #[test]
    fn test_model_sync() {
        let mut model = Model::new(1);
        model.add_root(0.0, 0.0);
        let (next_model_s, mut next_model_r) = channel::bounded(100, "updated-models");
        let mut model_sync = ModelSync::new(
            &model,
            100,
            &"test".to_string(),
            0.1,
            10,
            next_model_s,
            &vec![],
            5,
        );
        let packet_sender = model_sync.start_network(
            "tester".to_string(), vec!["s1".to_string(), "s2".to_string()], 8000);
        assert!(try_recv(&mut next_model_r).is_some());

        // no packet sent
        model_sync.sync_once();
        model_sync.sync_once();
        // assert_eq!(num_consecutive_err, 2);

        // send a good packet
        let packet = get_mock_packet(0, 0, 0.5, 1);
        assert!(try_recv(&mut next_model_r).is_none());
        packet_sender.send(packet).unwrap();
        model_sync.sync_once();
        // assert_eq!(num_consecutive_err, 0);
        assert!(try_recv(&mut next_model_r).is_some());

        // send many empty packets
        assert!(try_recv(&mut next_model_r).is_none());
        for _ in 0..10 {
            let packet = get_mock_packet(0, 0, 0.5, 0);
            packet_sender.send(packet).unwrap();
            model_sync.sync_once();
        }
        // assert_eq!(num_consecutive_err, 0);
        assert!(try_recv(&mut next_model_r).is_none());
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