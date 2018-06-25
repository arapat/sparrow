mod bitmap;
mod disk_buffer;
mod stratum;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::mpsc::SyncSender;
use chan::Receiver;

use bincode::serialize;

use commons::Example;
use commons::TFeature;
use commons::TLabel;
use labeled_data::LabeledData;

use super::ExampleWithScore;
use self::disk_buffer::DiskBuffer;
use self::stratum::Stratum;


type Block = Vec<ExampleWithScore>;
type InQueueSender = SyncSender<ExampleWithScore>;
type OutQueueReceiver = Receiver<ExampleWithScore>;

type HashMapSenders = HashMap<i8, InQueueSender>;
type HashMapReceiver = HashMap<i8, OutQueueReceiver>;


pub struct Strata {
    num_examples_per_block: usize,
    disk_buffer: Arc<RwLock<DiskBuffer>>,

    in_queues: Arc<RwLock<HashMapSenders>>,
    out_queues: Arc<RwLock<HashMapReceiver>>
}


macro_rules! unlock_read {
    ($lock:expr) => ($lock.read().unwrap());
}


macro_rules! unlock_write {
    ($lock:expr) => ($lock.write().unwrap());
}


impl Strata {
    pub fn new(num_examples: usize,
               feature_size: usize,
               num_examples_per_block: usize,
               disk_buffer_name: &str) -> Strata {
        let disk_buffer = get_disk_buffer(
            disk_buffer_name, feature_size, num_examples, num_examples_per_block);
        Strata {
            num_examples_per_block: num_examples_per_block,
            disk_buffer: get_locked(disk_buffer),
            in_queues: Arc::new(RwLock::new(HashMap::new())),
            out_queues: Arc::new(RwLock::new(HashMap::new()))
        }
    }

    pub fn get_in_queue(&self, index: i8) -> Option<InQueueSender> {
        if let Some(t) = unlock_read!(self.in_queues).get(&index) {
            Some(t.clone())
        } else {
            None
        }
    }

    pub fn get_out_queue(&self, index: i8) -> Option<OutQueueReceiver> {
        if let Some(t) = unlock_read!(self.out_queues).get(&index) {
            Some(t.clone())
        } else {
            None
        }
    }

    pub fn create(&mut self, index: i8) -> (InQueueSender, OutQueueReceiver) {
        if unlock_read!(self.in_queues).contains_key(&index) {
            (
                unlock_read!(self.in_queues).get(&index).unwrap().clone(),
                unlock_read!(self.out_queues).get(&index).unwrap().clone()
            )
        } else {
            let mut stratum = Stratum::new(self.num_examples_per_block, self.disk_buffer.clone());
            stratum.run();
            let in_queue = stratum.get_in_queue();
            let out_queue = stratum.get_out_queue();
            unlock_write!(self.in_queues).insert(index, in_queue.clone());
            unlock_write!(self.out_queues).insert(index, out_queue.clone());
            (in_queue, out_queue)
        }
    }
}


pub fn get_disk_buffer(filename: &str, feature_size: usize,
                       num_examples: usize, num_examples_per_block: usize) -> DiskBuffer {
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


fn get_locked<T>(t: T) -> Arc<RwLock<T>> {
    Arc::new(RwLock::new(t))
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
        let mut examples = vec![];
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
                sender.send(t.clone()).unwrap();
                examples.push(t);
            }
        }
        let mut answer = examples.iter();
        for i in 0..100 {
            for k in 0..10 {
                let retrieve = strata.get_out_queue(k as i8).unwrap().recv().unwrap();
                assert_eq!(*answer.next().unwrap(), retrieve);
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