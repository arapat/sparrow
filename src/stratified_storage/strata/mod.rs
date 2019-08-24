mod bitmap;
mod disk_buffer;
mod stratum;

use bincode::serialize;
use bincode::deserialize;

use std::vec::IntoIter;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;

use commons::ExampleWithScore;
use commons::channel::Sender;
use commons::channel::Receiver;
use super::Example;

use super::super::TFeature;
use super::super::TLabel;
use labeled_data::LabeledData;

use self::disk_buffer::DiskBuffer;
use self::stratum::Stratum;


type Block = Vec<ExampleWithScore>;
type ArcBlockIter = Arc<RwLock<IntoIter<ExampleWithScore>>>;
type QueueSender = Sender<ExampleWithScore>;
type QueueReceiver = Receiver<ExampleWithScore>;

type HashMapSenders = HashMap<i8, QueueSender>;
type HashMapReceiver = HashMap<i8, QueueReceiver>;
type HashMapBlockIters = HashMap<i8, ArcBlockIter>;

pub struct Strata {
    num_examples_per_block: usize,
    disk_buffer: Arc<RwLock<DiskBuffer>>,

    in_queues: Arc<RwLock<HashMapSenders>>,
    in_queues_receivers: HashMapReceiver,
    out_queues: Arc<RwLock<HashMapReceiver>>,
    out_block: HashMapBlockIters,
    sampler_state: Arc<RwLock<bool>>,
}


impl Strata {
    pub fn new(
        num_examples: usize,
        feature_size: usize,
        num_examples_per_block: usize,
        disk_buffer_name: &str,
        sampler_state: Arc<RwLock<bool>>,
        serialized_data: Option<Vec<u8>>,
    ) -> Strata {
        let (ser_num_examples_per_block, disk_buffer, data_in_queues) = {
            if serialized_data.is_some() {
                get_deserialize(serialized_data.unwrap())
            } else {
                (
                    num_examples_per_block,
                    get_disk_buffer(
                        disk_buffer_name, feature_size, num_examples, num_examples_per_block),
                    HashMap::new(),
                )
            }
        };
        assert_eq!(ser_num_examples_per_block, num_examples_per_block);
        let mut strata = Strata {
            num_examples_per_block: num_examples_per_block,
            disk_buffer:            Arc::new(RwLock::new(disk_buffer)),
            in_queues:              Arc::new(RwLock::new(HashMap::new())),
            in_queues_receivers:    HashMap::new(),
            out_queues:             Arc::new(RwLock::new(HashMap::new())),
            out_block:              HashMap::new(),
            sampler_state:          sampler_state,
        };
        for (key, vals) in data_in_queues {
            let (in_queue, _) = strata.create(key);
            for example in vals {
                in_queue.send(example);
            }
        }
        strata
    }

    pub fn get_in_queue(&self, index: i8) -> Option<QueueSender> {
        if let Some(t) = self.in_queues.read().unwrap().get(&index) {
            Some(t.clone())
        } else {
            None
        }
    }

    pub fn get_out_queue(&self, index: i8) -> Option<QueueReceiver> {
        if let Some(t) = self.out_queues.read().unwrap().get(&index) {
            Some(t.clone())
        } else {
            None
        }
    }

    pub fn create(&mut self, index: i8) -> (QueueSender, QueueReceiver) {
        let (mut in_queues, mut out_queues) =
            (self.in_queues.write().unwrap(), self.out_queues.write().unwrap());
        if in_queues.contains_key(&index) {
            // Other process have created the stratum before this process secures the writing lock
            (in_queues[&index].clone(), out_queues[&index].clone())
        } else {
            // Each stratum will create two threads for writing in and reading out examples
            // TODO: create a systematic approach to manage stratum threads
            let stratum = Stratum::new(
                index, self.num_examples_per_block, self.disk_buffer.clone(),
                self.sampler_state.clone());
            let (in_queue, out_queue) = (stratum.in_queue_s.clone(), stratum.out_queue_r.clone());
            in_queues.insert(index, in_queue.clone());
            self.in_queues_receivers.insert(index, stratum.in_queue_r.clone());
            out_queues.insert(index, out_queue.clone());
            self.out_block.insert(index, stratum.out_block.clone());
            (in_queue, out_queue)
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        let ser_disk_buffer = {
            let disk_buffer = self.disk_buffer.read().unwrap();
            disk_buffer.serialize()
        };
        info!("Snapshot of the disk buffer has been taken.");
        let out_queues = self.out_queues.read().unwrap();
        let mut data_in_queues = HashMap::new();
        for key in out_queues.keys() {
            let in_queue: QueueReceiver = self.in_queues_receivers[key].clone();
            let out_queue: QueueReceiver = out_queues[key].clone();

            let mut ret = Vec::with_capacity(in_queue.len() + out_queue.len());
            let mut example = in_queue.try_recv();
            while example.is_some() {
                ret.push(example.unwrap());
                example = in_queue.try_recv();
            }
            {
                let mut out_block_ptr = self.out_block[key].write().unwrap();
                example = out_block_ptr.next();
                while example.is_some() {
                    ret.push(example.unwrap());
                    example = out_block_ptr.next();
                }
            }
            example = out_queue.try_recv();
            while example.is_some() {
                ret.push(example.unwrap());
                example = out_queue.try_recv();
            }
            data_in_queues.insert(*key, ret);
        }
        info!("Snapshot of the in/out queues has been taken.");
        let data: (usize, Vec<u8>, HashMap<i8, Vec<ExampleWithScore>>) =
            (self.num_examples_per_block, ser_disk_buffer, data_in_queues);
        serialize(&data).unwrap()
    }
}


fn get_deserialize(data: Vec<u8>) -> (usize, DiskBuffer, HashMap<i8, Vec<ExampleWithScore>>) {
    let (num_examples_per_block, ser_disk_buffer, data_in_queues) = deserialize(&data).unwrap();
    let mut disk_buffer: DiskBuffer = deserialize(ser_disk_buffer).unwrap();
    disk_buffer.init_file();
    (num_examples_per_block, disk_buffer, data_in_queues)
}


pub fn get_disk_buffer(
    filename: &str,
    feature_size: usize,
    num_examples: usize,
    num_examples_per_block: usize,
) -> DiskBuffer {
    let num_disk_block = (num_examples + num_examples_per_block - 1) / num_examples_per_block;
    let block_size = get_block_size(feature_size, num_examples_per_block);
    DiskBuffer::new(filename, block_size, num_disk_block)
}


fn get_block_size(feature_size: usize, num_examples_per_block: usize) -> usize {
    let example: Example = LabeledData::new(vec![0 as TFeature; feature_size], -1 as TLabel);
    let example_with_score: ExampleWithScore = (example, (0.0, 0));
    let block: Block = vec![example_with_score; num_examples_per_block];
    let serialized_block: Vec<u8> = serialize(&block).unwrap();
    serialized_block.len()
}


#[cfg(test)]
mod tests {
    use std::fs::remove_file;

    use labeled_data::LabeledData;
    use commons::ExampleWithScore;
    use super::Strata;
    use ::TFeature;

    #[test]
    fn test_strata() {
        let filename = "unittest-strata.bin";
        let mut strata = Strata::new(1000, 3, 10, filename);
        for i in 0..100 {
            for k in 0..10 {
                let t = get_example(vec![0 as TFeature, i as TFeature, k as TFeature]);
                let mut sender = {
                    if let Some(t) = strata.get_in_queue(k as i8) {
                        t
                    } else {
                        let (sender, _) = strata.create(k as i8);
                        sender
                    }
                };
                sender.send(t.clone());
            }
        }
        for _ in 0..100 {
            for k in 0..10 {
                let retrieve = strata.get_out_queue(k as i8).unwrap().recv().unwrap();
                assert_eq!(k as TFeature, retrieve.0.feature[2]);
            }
        }
        remove_file(filename).unwrap();
    }

    fn get_example(features: Vec<TFeature>) -> ExampleWithScore {
        let label: i8 = -1;
        let example = LabeledData::new(features, label);
        (example, (1.0, 0))
    }
}