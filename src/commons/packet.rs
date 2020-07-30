use commons::tree::UpdateList;
use commons::Model;


#[derive(Clone, PartialEq)]
pub enum BoosterState {
    IDLE,
    STOPPING,
    RUNNING,
}


#[derive(Debug)]
pub enum UpdatePacketType {
    Accept,
    Empty,
}


#[derive(Serialize, Deserialize, Debug)]
pub struct TaskPacket {
    pub packet_id: usize,
    pub model: Model,
    pub gamma: f32,
    pub expand_node: usize,
    pub new_sample_version: Option<usize>,
}


impl TaskPacket {
    pub fn new(
        packet_id: usize,
        model: Model,
        gamma: f32,
        expand_node: usize,
        new_sample_version: Option<usize>,
    ) -> TaskPacket {
        TaskPacket {
            packet_id: packet_id,
            model: model,
            gamma: gamma,
            expand_node: expand_node,
            new_sample_version: None,
        }
    }
}


#[derive(Serialize, Deserialize, Debug)]
pub struct UpdatePacket {
    pub packet_id: usize,
    pub updates: UpdateList,

    pub task: TaskPacket,
    pub sample_version: usize,
    pub ess: f32,
}


impl UpdatePacket {
    pub fn new(
        packet_id: usize,
        updates: UpdateList,
        task: TaskPacket,
        sample_version: usize,
        ess: f32,
    ) -> UpdatePacket {
        // TODO: remove sigs
        // let this_model_sig = machine_name.clone() + "_" + &final_model_size.to_string();
        // let packet_sig = format!("pac_{}_{}", this_model_sig, packet_counter);
        UpdatePacket {
            packet_id: packet_id,
            updates: updates,
            task: task,
            sample_version: sample_version,
            ess: ess,
        }
    }

    pub fn get_packet_type(&self, min_ess: f32) -> UpdatePacketType {
        if self.updates.size == 0 {
            // Empty packets
            debug!("model_manager, packet, empty");
            UpdatePacketType::Empty
        } else {
            debug!("model_manager, packet, accept");
            UpdatePacketType::Accept
        }
    }
}