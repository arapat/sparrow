use commons::model::Model;
use commons::tree::Tree;


#[derive(Clone, PartialEq)]
pub enum BoosterState {
    IDLE,
    STOPPING,
    RUNNING,
}


#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum UpdatePacketType {
    Accept,
    Empty,
    BaseVersionMismatch,
    Unset,
}


#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
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

    pub fn set_model(&mut self, model: Model) {
        self.model = Some(model);
    }

    pub fn set_gamma(&mut self, gamma: f32) {
        self.gamma = Some(gamma);
    }

    pub fn set_packet_id(&mut self, packet_id: usize) {
        self.packet_id = packet_id;
    }

    pub fn set_expand_node(&mut self, expand_node: Option<usize>) {
        self.expand_node = expand_node;
    }

    pub fn set_sample_version(&mut self, sample_version: usize) {
        self.new_sample_version = Some(sample_version);
    }

    pub fn equals(&self, other: &TaskPacket) -> bool {
        self.model == other.model && self.gamma == other.gamma && (
            other.expand_node.is_none() || self.expand_node == other.expand_node
        )
    }

    pub fn clone_with_expand(&self, other: &TaskPacket) -> TaskPacket {
        let mut ret = self.clone();
        if ret.expand_node.is_none() {
            ret.expand_node = other.expand_node.clone();
        }
        if ret.new_sample_version.is_none() {
            ret.new_sample_version = other.new_sample_version.clone();
        }
        ret
    }
}


#[derive(Serialize, Deserialize, Debug)]
pub struct UpdatePacket {
    pub packet_id: usize,
    pub update_tree: Option<Tree>,
    pub base_size: usize,

    pub task: TaskPacket,
    pub sample_version: usize,
    pub ess: f32,

    pub packet_type: UpdatePacketType,
}


impl UpdatePacket {
    pub fn new(
        update_tree: Option<Tree>,
        base_size: usize,
        task: TaskPacket,
        sample_version: usize,
        ess: f32,
    ) -> UpdatePacket {
        UpdatePacket {
            packet_id: 0,
            update_tree: update_tree,
            base_size: base_size,
            task: task,
            sample_version: sample_version,
            ess: ess,
            packet_type: UpdatePacketType::Unset,
        }
    }

    pub fn set_packet_id(&mut self, packet_id: usize) {
        self.packet_id = packet_id;
    }

    pub fn set_packet_type(&mut self, curr_model_size: usize) {
        self.packet_type = {
            if self.base_size != curr_model_size {
                debug!("model_manager, packet, base version mismatch");
                UpdatePacketType::BaseVersionMismatch
            } else if self.update_tree.is_none() {
                debug!("model_manager, packet, empty");
                UpdatePacketType::Empty
            } else {
                debug!("model_manager, packet, accept");
                UpdatePacketType::Accept
            }
        };
    }
}