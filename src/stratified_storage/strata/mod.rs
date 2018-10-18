mod bitmap;
mod disk_buffer;
mod stratum;

use crossbeam_channel as channel;
use bincode::serialize;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use self::channel::Sender;
use self::channel::Receiver;

use commons::ExampleWithScore;
use super::Example;

use super::super::TFeature;
use super::super::TLabel;
use labeled_data::LabeledData;

use self::disk_buffer::DiskBuffer;
use self::stratum::Stratum;


type Block = Vec<ExampleWithScore>;
type InQueueSender = Sender<ExampleWithScore>;
type OutQueueReceiver = Receiver<ExampleWithScore>;

type HashMapSenders = HashMap<i8, InQueueSender>;
type HashMapReceiver = HashMap<i8, OutQueueReceiver>;


pub struct Strata {
    num_examples_per_block: usize,
    disk_buffer: Arc<RwLock<DiskBuffer>>,

    in_queues: Arc<RwLock<HashMapSenders>>,
    out_queues: Arc<RwLock<HashMapReceiver>>
}


impl Strata {
    pub fn new(
        num_examples: usize,
        feature_size: usize,
        num_examples_per_block: usize,
        disk_buffer_name: &str,
    ) -> Strata {
        let disk_buffer = get_disk_buffer(
            disk_buffer_name, feature_size, num_examples, num_examples_per_block);
        Strata {
            num_examples_per_block: num_examples_per_block,
            disk_buffer: Arc::new(RwLock::new(disk_buffer)),
            in_queues: Arc::new(RwLock::new(HashMap::new())),
            out_queues: Arc::new(RwLock::new(HashMap::new()))
        }
    }

    pub fn get_in_queue(&self, index: i8) -> Option<InQueueSender> {
        if let Some(t) = self.in_queues.read().unwrap().get(&index) {
            Some(t.clone())
        } else {
            None
        }
    }

    pub fn get_out_queue(&self, index: i8) -> Option<OutQueueReceiver> {
        if let Some(t) = self.out_queues.read().unwrap().get(&index) {
            Some(t.clone())
        } else {
            None
        }
    }

    pub fn create(&mut self, index: i8) -> (InQueueSender, OutQueueReceiver) {
        let (mut in_queues, mut out_queues) =
            (self.in_queues.write().unwrap(), self.out_queues.write().unwrap());
        if in_queues.contains_key(&index) {
            // Other process have created the stratum before this process secures the writing lock
            (in_queues[&index].clone(), out_queues[&index].clone())
        } else {
            // Each stratum will create two threads for writing in and reading out examples
            // TODO: create a systematic approach to manage stratum threads
            let stratum = Stratum::new(self.num_examples_per_block, self.disk_buffer.clone());
            let (in_queue, out_queue) = (stratum.in_queue_s.clone(), stratum.out_queue_r.clone());
            in_queues.insert(index, in_queue.clone());
            out_queues.insert(index, out_queue.clone());
            (in_queue, out_queue)
        }
    }
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
    let example: Example = LabeledData::new(vec![0 as TFeature; feature_size], 0 as TLabel);
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

    #[test]
    fn test_strata() {
        let filename = "unittest-strata.bin";
        let mut strata = Strata::new(1000, 3, 10, filename);
        for i in 0..100 {
            for k in 0..10 {
                let t = get_example(vec![0, i, k]);
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
                assert_eq!(k, retrieve.0.feature[2]);
            }
        }
        remove_file(filename).unwrap();
    }

    fn get_example(features: Vec<u8>) -> ExampleWithScore {
        let label: u8 = 0;
        let example = LabeledData::new(features, label);
        (example, (1.0, 0))
    }
}