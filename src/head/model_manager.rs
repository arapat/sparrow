use commons::packet::UpdatePacket;
use commons::packet::UpdatePacketType;
use commons::performance_monitor::PerformanceMonitor;

use commons::persistent_io::write_model;
use head::model_with_version::ModelWithVersion;


pub struct ModelManager {
    model: ModelWithVersion,
    model_ts: f32,

    _performance_mon: PerformanceMonitor,
    _last_logging_ts: f32,
}


impl ModelManager {
    pub fn new(init_tree: &ModelWithVersion) -> ModelManager {
        ModelManager {
            model: init_tree.clone(),
            model_ts: 0.0,

            _performance_mon: PerformanceMonitor::new(),
            _last_logging_ts: 0.0,
        }
    }

    pub fn handle_packet(
        &mut self, source_ip: &String, packet: &mut UpdatePacket,
    ) -> ModelWithVersion {
        if packet.get_packet_type() == UpdatePacketType::Accept {
            self.model_ts = self._performance_mon.get_duration();
            self.update_model(&source_ip, &packet);
            self.print_log();
        }
        self.model.clone()
    }

    fn broadcast_model(&mut self, is_model_updated: bool) {
        // callback TODO: fix upload model
        // let is_upload_success = upload_model(
        //     &self.model.model, &self.model.model_sig, self.gamma.gamma, &self.exp_name);
        if is_model_updated {
            write_model(&self.model.model, self.model_ts, true);
        }
        // debug!("model_manager, upload model, {}, {}",
        //         is_upload_success, self.model.model_sig);
    }


    fn update_model(&mut self, last_update_from: &String, packet: &UpdatePacket) {
        self.model.update(packet.update_tree.clone(), last_update_from);
        self.broadcast_model(true);
        debug!("model_manager, new updates, {}", self.model.size());
    }


    fn print_log(&mut self) {
        if self._performance_mon.get_duration() - self._last_logging_ts >= 10.0 {
            self.model.print_log();
            self._last_logging_ts = self._performance_mon.get_duration();
        }
    }
}


/*
#[cfg(test)]
mod tests {
    use super::ModelManager;
    use commons::model::Model;
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
        packet_sender.set_input("./debug.txt");
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
*/