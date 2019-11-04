use std::cmp::max;

use commons::packet::Packet;
use commons::packet::PacketType;


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
    rejected_packets:        usize,

    accept_root_packets:     usize,
    accept_nonroot_packets:  usize,
    empty_root_packets:      usize,
    empty_nonroot_packets:   usize,
    rejected_packets_model:  usize,
    rejected_packets_sample: usize,

    avg_accept_nonroot_rate: f32,
    threshold:               f32,

    last_nonroot_condition:  UpdateSpeed,
    pub curr_nonroot_condition:  UpdateSpeed,

    num_packs:               Vec<usize>,
    num_rej_packs:           Vec<usize>,
    num_acc_nonroot_packs:   Vec<usize>,
    num_empty_packs:         Vec<usize>,
    pub num_machines:            usize,
}


impl PacketStats {
    pub fn new(num_machines: usize) -> PacketStats {
        PacketStats {
            total_packets:           0,
            empty_packets:           0,
            accept_packets:          0,
            rejected_packets:          0,

            accept_root_packets:     0,
            accept_nonroot_packets:  0,
            empty_root_packets:      0,
            empty_nonroot_packets:   0,
            rejected_packets_model:  0,
            rejected_packets_sample: 0,

            avg_accept_nonroot_rate: 0.5,
            threshold:               THRESHOLD,

            last_nonroot_condition:  UpdateSpeed::Okay,
            curr_nonroot_condition:  UpdateSpeed::Okay,

            num_packs:               vec![0; num_machines],
            num_rej_packs:           vec![0; num_machines],
            num_acc_nonroot_packs:   vec![0; num_machines],
            num_empty_packs:         vec![0; num_machines],
            num_machines:            num_machines,
        }
    }

    pub fn handle_new_packet(&mut self, packet: &Packet, packet_type: &PacketType) {
        self.total_packets += 1;
        let machine_id = packet.source_machine_id;
        self.num_packs[machine_id] += 1;
        match packet_type {
            PacketType::AcceptRoot => {
                self.accept_packets          += 1;
                self.accept_root_packets     += 1;
            },
            PacketType::AcceptNonroot => {
                self.accept_packets          += 1;
                self.accept_nonroot_packets  += 1;
                self.num_acc_nonroot_packs[machine_id] += 1;
            },
            PacketType::EmptyRoot => {
                self.empty_packets           += 1;
                self.empty_root_packets      += 1;
                self.num_empty_packs[machine_id] += 1;
            },
            PacketType::EmptyNonroot => {
                self.empty_packets           += 1;
                self.empty_nonroot_packets   += 1;
                self.num_empty_packs[machine_id] += 1;
            },
            PacketType::RejectSample => {
                self.rejected_packets        += 1;
                self.rejected_packets_sample += 1;
                self.num_rej_packs[machine_id] += 1;
            },
            PacketType::RejectBaseModel => {
                self.rejected_packets        += 1;
                self.rejected_packets_model  += 1;
                self.num_rej_packs[machine_id] += 1;
            },
        };
    }

    pub fn update_condition(&mut self) {
        self.last_nonroot_condition = self.curr_nonroot_condition.clone();
        let (rate, cond) = get_condition_updates(
            self.accept_nonroot_packets, self.empty_nonroot_packets, self.avg_accept_nonroot_rate,
            self.threshold,
        );
        self.avg_accept_nonroot_rate = rate;
        self.curr_nonroot_condition = cond;
    }

    pub fn is_triggered(&self) -> bool {
        self.total_packets >= max(10, self.num_machines * 2)
    }

    pub fn is_nonroot_same_trend(&self) -> bool {
        self.curr_nonroot_condition != UpdateSpeed::Okay &&
            self.curr_nonroot_condition == self.last_nonroot_condition
    }

    pub fn is_nonroot_opposite_trend(&self) -> bool {
        self.curr_nonroot_condition != UpdateSpeed::Okay &&
            self.last_nonroot_condition != UpdateSpeed::Okay &&
            self.curr_nonroot_condition != self.last_nonroot_condition
    }

    pub fn reset(&mut self) {
        self.print_log();

        self.total_packets = 0;
        self.empty_packets = 0;
        self.accept_packets = 0;
        self.rejected_packets = 0;

        self.accept_root_packets = 0;
        self.accept_nonroot_packets = 0;
        self.empty_root_packets = 0;
        self.empty_nonroot_packets = 0;
        self.rejected_packets_model = 0;
        self.rejected_packets_sample = 0;

        self.num_packs.iter_mut()
            .for_each(|t| *t = 0);
        self.num_rej_packs.iter_mut()
            .for_each(|t| *t = 0);
        self.num_acc_nonroot_packs.iter_mut()
            .for_each(|t| *t = 0);
        self.num_empty_packs.iter_mut()
            .for_each(|t| *t = 0);
    }

    fn print_log(&self) {
        debug!("model_manager, packet stats, {}",
            (vec![
                self.total_packets.to_string(),
                self.empty_packets.to_string(),
                self.accept_packets.to_string(),
                self.rejected_packets.to_string(),

                self.accept_root_packets.to_string(),
                self.accept_nonroot_packets.to_string(),
                self.empty_root_packets.to_string(),
                self.empty_nonroot_packets.to_string(),
                self.rejected_packets_model.to_string(),
                self.rejected_packets_sample.to_string(),

                self.avg_accept_nonroot_rate.to_string(),
                format!("{:?}", self.curr_nonroot_condition),

                vec_to_string(&self.num_packs),
                vec_to_string(&self.num_rej_packs),
                vec_to_string(&self.num_acc_nonroot_packs),
                vec_to_string(&self.num_empty_packs),
            ]).join(", ")
        );
    }
}


fn get_condition_updates(
    accept: usize, empty: usize, old_avg_rate: f32, threshold: f32,
) -> (f32, UpdateSpeed) {
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
    (new_avg_rate, condition)
}


fn vec_to_string(vec: &Vec<usize>) -> String {
    let s: Vec<String> = vec.iter().map(|t| t.to_string()).collect();
    s.join(", ")
}
