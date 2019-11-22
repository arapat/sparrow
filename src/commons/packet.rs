use std::sync::Arc;
use std::sync::RwLock;

use commons::tree::UpdateList;


#[derive(Debug)]
pub enum PacketType {
    AcceptRoot,
    AcceptNonroot,
    EmptyRoot,
    EmptyNonroot,
    SmallEffSize,
    RejectSample,
    RejectBaseModel,
}


#[derive(Serialize, Deserialize, Debug)]
pub struct Packet {
    pub packet_signature: String,
    pub source_machine: String,
    pub source_machine_id: usize,
    pub node_id: usize,
    pub updates: UpdateList,
    pub gamma: f32,
    pub sample_version: usize,
    pub ess: f32,
    pub base_model_signature: String,
    pub this_model_signature: String,
}


impl Packet {
    pub fn new(
        machine_name: &String,
        machine_id: usize,
        packet_counter: usize,
        node_id: usize,
        final_model_size: usize,
        updates: UpdateList,
        gamma: f32,
        ess: f32,
        sample_version: usize,
        base_model_sig: String,
    ) -> Packet {
        let this_model_sig = machine_name.clone() + "_" + &final_model_size.to_string();
        let packet_sig = format!("pac_{}_{}", this_model_sig, packet_counter);
        Packet {
            packet_signature: packet_sig,
            source_machine: machine_name.clone(),
            source_machine_id: machine_id,
            node_id: node_id,
            updates: updates,
            gamma: gamma,
            sample_version: sample_version,
            ess: ess,
            base_model_signature: base_model_sig,
            this_model_signature: this_model_sig,
        }
    }

    pub fn get_packet_type(
        &self, sampler_sample_version: &Arc<RwLock<usize>>, sampler_model_version: &String,
        min_ess: f32,
    ) -> PacketType {
        // Ignore any claims made on a very small effective sample
        if self.ess < min_ess {
            debug!("model_manager, packet, empty small ess, {}, {}, {}",
                    self.source_machine_id, self.node_id, self.ess);
            return PacketType::SmallEffSize;
        }

        // Empty packets
        if self.updates.size == 0 {
            if self.node_id == 0 {
                debug!("model_manager, packet, empty root, {}, {}, {}",
                        self.source_machine_id, self.node_id, self.ess);
                return PacketType::EmptyRoot;
            }
            debug!("model_manager, packet, empty nonroot, {}, {}, {}",
                    self.source_machine_id, self.node_id, self.ess);
            return PacketType::EmptyNonroot;
        }

        // Reject packets
        if self.base_model_signature != *sampler_model_version {
            debug!("model_manager, reject for base model mismatch, {}, {}, {}",
                    self.base_model_signature, sampler_model_version, self.packet_signature);
            debug!("model_manager, packet, reject model, {}, {}",
                    self.source_machine_id, self.node_id);
            return PacketType::RejectBaseModel;
        }
        let current_version: usize = {
            *(sampler_sample_version.read().unwrap())
        };
        if self.sample_version != current_version {
            debug!("model_manager, reject for sample version mismatch, {}, {}, {}",
                    self.sample_version, current_version, self.packet_signature);
            debug!("model_manager, packet, reject sample, {}, {}",
                    self.source_machine_id, self.node_id);
            return PacketType::RejectSample;
        }

        // Accept packets
        if self.node_id == 0 {
            debug!("model_manager, packet, accept root, {}, {}, {}",
                    self.source_machine_id, self.node_id, self.ess);
            return PacketType::AcceptRoot;
        }
        debug!("model_manager, packet, accept nonroot, {}, {}, {}",
                self.source_machine_id, self.node_id, self.ess);
        PacketType::AcceptNonroot
    }
}
