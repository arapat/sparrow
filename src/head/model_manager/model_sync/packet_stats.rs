use std::cmp::max;
use std::collections::HashMap;

use commons::packet::UpdatePacket;
use commons::packet::UpdatePacketType;


// TODO: move the parameters into the config
const THRESHOLD: f32 = 0.1;
// Smoothing factor
const ETA: f32 = 0.1;


#[derive(Debug, Clone, PartialEq)]
pub enum UpdateSpeed {
    Okay,
    TooFast,
    TooSlow,
}



pub struct PacketStats {
    total_packets:           usize,
    empty_packets:           usize,
    accept_packets:          usize,
    small_ess_packets:       usize,

    pub avg_accept_rate:  f32,
    pub last_accept_rate: f32,
    threshold:                    f32,

    last_condition:      UpdateSpeed,
    pub curr_condition:  UpdateSpeed,

    num_packs:           HashMap<String, usize>,
    num_acc_packs:       HashMap<String, usize>,
    num_empty_packs:     HashMap<String, usize>,
    pub num_machines:    usize,
}


impl PacketStats {
    pub fn new(num_machines: usize) -> PacketStats {
        PacketStats {
            total_packets:           0,
            empty_packets:           0,
            accept_packets:          0,
            small_ess_packets:       0,

            avg_accept_rate:         0.5,
            last_accept_rate:        0.5,
            threshold:               THRESHOLD,

            last_condition:  UpdateSpeed::Okay,
            curr_condition:  UpdateSpeed::Okay,

            num_packs:       HashMap::new(),
            num_acc_packs:   HashMap::new(),
            num_empty_packs: HashMap::new(),
            num_machines:    num_machines,
        }
    }

    pub fn handle_new_packet(
        &mut self, source_ip: &String, packet: &UpdatePacket, packet_type: &UpdatePacketType,
    ) {
        self.total_packets += 1;
        let num_packs = self.num_packs.entry(source_ip.clone()).or_insert(0);
        *num_packs += 1;
        match packet_type {
            UpdatePacketType::Accept => {
                self.accept_packets           += 1;
                let num_acc_packs = self.num_acc_packs.entry(source_ip.clone()).or_insert(0);
                *num_acc_packs   += 1;
            },
            UpdatePacketType::Empty => {
                self.empty_packets              += 1;
                let num_empty_packs = self.num_empty_packs.entry(source_ip.clone()).or_insert(0);
                *num_empty_packs   += 1;
            },
        };
        self.update_condition();
    }

    fn update_condition(&mut self) {
        self.last_condition = self.curr_condition.clone();
        let (avg_rate, last_rate, cond) = get_condition_updates(
            self.accept_packets, self.empty_packets, self.avg_accept_rate, self.threshold,
        );
        self.avg_accept_rate  = avg_rate;
        self.last_accept_rate = last_rate;
        self.curr_condition   = cond;
    }

    pub fn got_sufficient_packages(&self) -> bool {
        self.total_packets >= max(10, self.num_machines * 2)
    }

    pub fn is_same_trend(&self) -> bool {
        self.curr_condition != UpdateSpeed::Okay &&
            self.curr_condition == self.last_condition
    }

    pub fn is_opposite_trend(&self) -> bool {
        self.curr_condition != UpdateSpeed::Okay &&
            self.last_condition != UpdateSpeed::Okay &&
            self.curr_condition != self.last_condition
    }

    pub fn reset(&mut self) {
        self.total_packets = 0;
        self.empty_packets = 0;
        self.accept_packets = 0;
        self.small_ess_packets = 0;

        self.num_packs.iter_mut()
            .for_each(|(_, val)| *val = 0);
        self.num_acc_packs.iter_mut()
            .for_each(|(_, val)| *val = 0);
        self.num_empty_packs.iter_mut()
            .for_each(|(_, val)| *val = 0);
    }

    pub fn print_log(&self) {
        debug!("model_manager, packet stats, status, {}",
            (vec![
                self.total_packets.to_string(),
                self.empty_packets.to_string(),
                self.accept_packets.to_string(),
                self.small_ess_packets.to_string(),

                self.avg_accept_rate.to_string(),
                self.last_accept_rate.to_string(),
                format!("{:?}", self.curr_condition),

                map_to_string(&self.num_packs),
                map_to_string(&self.num_acc_packs),
                map_to_string(&self.num_empty_packs),
            ]).join(", ")
        );
    }
}


fn get_condition_updates(
    accept: usize, empty: usize, old_avg_rate: f32, threshold: f32,
) -> (f32, f32, UpdateSpeed) {
    let accept_rate = (accept as f32) / (max(1, accept + empty) as f32);
    let new_avg_rate = (1.0 - ETA) * old_avg_rate + ETA * accept_rate;
    let condition = {
        if new_avg_rate <= threshold {
            UpdateSpeed::TooSlow
        } else if new_avg_rate >= 1.0 - threshold {
            UpdateSpeed::TooFast
        } else {
            UpdateSpeed::Okay
        }
    };
    (new_avg_rate, accept_rate, condition)
}


fn map_to_string(vec: &HashMap<String, usize>) -> String {
    let s: Vec<String> = vec.iter().map(|(key, val)| format!("{}({})", key, val)).collect();
    s.join(", ")
}
