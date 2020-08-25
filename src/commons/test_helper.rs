use std::sync::mpsc;
use rand;
use rand::Rng;

use TFeature;
use commons::ExampleWithScore;
use commons::labeled_data::LabeledData;
use commons::packet::TaskPacket;
use commons::packet::UpdatePacket;
use commons::tree::Tree;


pub fn get_synthetic_example(features: Vec<TFeature>, label: i8, score: f32) -> ExampleWithScore {
    let example = LabeledData::new(features, label);
    (example, (score, 0))
}


pub fn get_n_random_examples(n: usize, num_features: usize) -> Vec<ExampleWithScore> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| {
        let features: Vec<TFeature> = (0..num_features).map(|_| { rng.gen::<TFeature>() })
                                                       .collect();
        let label: i8 = rng.gen_range(0, 1);
        get_synthetic_example(features, label, 0.0)
    }).collect()
}


pub fn get_mock_packet(
    _machine_id: usize, _node_id: usize, _gamma: f32, packet_size: usize,
) -> UpdatePacket {
    let ess = 0.5;
    let mut update_tree = Tree::new(1);
    let task_packet = TaskPacket::new();
    UpdatePacket::new(update_tree, 0, task_packet, 0, ess)
}


pub fn start_network_only_recv<T: 'static + Send>(
    _name: &str, _remote_ips: &Vec<String>, _port: u16, _data_remote: mpsc::Sender<T>,
) -> Result<(), &'static str> {
    Ok(())
}


pub fn start_network_only_send<T: 'static + Send>(
    _name: &str, _port: u16, _data_remote: mpsc::Receiver<T>,
) -> Result<(), &'static str> {
    Ok(())
}
