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


#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct TaskPacket {
    pub packet_id: usize,
    pub model: Option<Model>,
    pub gamma: Option<f32>,
    pub expand_node: Option<usize>,
    pub new_sample_version: Option<usize>,
}


impl TaskPacket {
    pub fn new() -> TaskPacket {
        TaskPacket {
            packet_id: 0,
            model: None,
            gamma: None,
            expand_node: None,
            new_sample_version: None,
        }
    }

    pub fn set_packet_id(&mut self, packet_id: usize) {
        self.packet_id = packet_id;
    }

    pub fn set_model(&mut self, model: Model) {
        self.model = Some(model);
    }

    pub fn set_gamma(&mut self, gamma: f32) {
        self.gamma = Some(gamma);
    }

    pub fn set_expand_node(&mut self, expand_node: usize) {
        self.expand_node = Some(expand_node);
    }

    pub fn set_sample_version(&mut self, sample_version: usize) {
        self.new_sample_version = Some(sample_version);
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