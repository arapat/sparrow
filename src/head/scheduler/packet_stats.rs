use std::collections::HashMap;

use commons::packet::UpdatePacketType;


// TODO: move the parameters into the config
const THRESHOLD: f32 = 0.1;


#[derive(Debug, Clone, PartialEq)]
pub enum UpdateSpeed {
    Okay,
    TooFast,
    TooSlow,
}


pub struct PacketStats {
    total_packets:       usize,
    packs_queue:         Vec<Option<UpdatePacketType>>,
    packs_next:          usize,

    last_condition:      UpdateSpeed,
    pub curr_condition:  UpdateSpeed,

    num_packs:           HashMap<String, usize>,
    num_packs_types:     HashMap<String, HashMap<UpdatePacketType, usize>>,
    pub num_machines:    usize,
}


impl PacketStats {
    pub fn new(num_machines: usize, packs_queue_size: usize) -> PacketStats {
        PacketStats {
            total_packets:   0,
            packs_queue:     vec![None; packs_queue_size],
            packs_next:      0,

            last_condition:  UpdateSpeed::Okay,
            curr_condition:  UpdateSpeed::Okay,

            num_packs:       HashMap::new(),
            num_packs_types: HashMap::new(),
            num_machines:    num_machines,
        }
    }

    pub fn handle_new_packet(&mut self, source_ip: &String, packet_type: &UpdatePacketType) {
        self.total_packets += 1;
        let num_packs = self.num_packs.entry(source_ip.clone()).or_insert(0);
        *num_packs += 1;
        if *packet_type == UpdatePacketType::Accept || *packet_type == UpdatePacketType::Empty {
            self.packs_queue[self.packs_next] = Some(packet_type.clone());
            self.packs_next = (self.packs_next + 1) % self.packs_queue.len();
        }
        let ip_num_packs_types =
            self.num_packs_types.entry(source_ip.clone()).or_insert(HashMap::new());
        let num_packs_types = (*ip_num_packs_types).entry(packet_type.clone()).or_insert(0);
        *num_packs_types += 1;
    }

    pub fn update_condition(&mut self) {
        let num_accept = self.packs_queue.iter()
                             .filter(|pack_type| (**pack_type) == Some(UpdatePacketType::Accept))
                             .count();
        let accept_rate = (num_accept as f32) / (self.packs_queue.len() as f32);
        self.last_condition = self.curr_condition.clone();
        self.curr_condition = {
            if accept_rate <= 0.0 {  // THRESHOLD {
                UpdateSpeed::TooSlow
            } else if accept_rate >= 1.0 - THRESHOLD {
                UpdateSpeed::TooFast
            } else {
                UpdateSpeed::Okay
            }
        };
        debug!("model_manager, packet stats, update condition, {}, {}", num_accept, accept_rate);
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

    pub fn print_log(&self) {
        let num_packs: Vec<String> = self.num_packs_types.iter()
                                         .map(|(key, val)| {
                                             format!("{}({})", key, map_to_string(val))
                                         }).collect();
        let num_packs = num_packs.join(", ");
        debug!("model_manager, packet stats, status, {}",
            (vec![
                self.total_packets.to_string(),

                format!("{:?}", self.curr_condition),

                map_to_string(&self.num_packs),
                num_packs,
            ]).join(", ")
        );
    }
}


fn map_to_string<T: std::fmt::Debug>(vec: &HashMap<T, usize>) -> String {
    let s: Vec<String> = vec.iter().map(|(key, val)| format!("{:?}({})", key, val)).collect();
    s.join(", ")
}