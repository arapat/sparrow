use std::sync::mpsc;
use std::thread::sleep;
use std::thread::spawn;
use bincode::serialize;
use bincode::deserialize;
use chan;


use std::collections::vec_deque::VecDeque;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;
use std::time::Duration;

use super::Block;
use super::ExampleWithScore;
use super::InQueueSender;
use super::OutQueueReceiver;
use super::disk_buffer::DiskBuffer;


pub struct Stratum {
    num_examples_per_block: usize,
    disk_buffer: Arc<RwLock<DiskBuffer>>,

    in_queue_in: Option<InQueueSender>,
    out_queue_out: Option<OutQueueReceiver>
}


macro_rules! unlock_write {
    ($lock:expr) => ($lock.write().unwrap());
}


impl Stratum {
    pub fn new(num_examples_per_block: usize, disk_buffer: Arc<RwLock<DiskBuffer>>) -> Stratum {
        Stratum {
            num_examples_per_block: num_examples_per_block,
            disk_buffer: disk_buffer,

            in_queue_in: None,
            out_queue_out: None
        }
    }

    pub fn get_in_queue(&mut self) -> InQueueSender {
        let mut t = get_clone(&mut self.in_queue_in);
        while t.is_none() {
            t = get_clone(&mut self.in_queue_in);
        }
        t.unwrap()
    }

    pub fn get_out_queue(&mut self) -> OutQueueReceiver {
        let mut t = get_clone(&mut self.out_queue_out);
        while t.is_none() {
            t = get_clone(&mut self.out_queue_out);
        }
        t.unwrap()
    }

    pub fn run(&mut self) {
        let (slot_in, slot_out) = mpsc::channel();
        let (in_queue_in, in_queue_out) = mpsc::sync_channel(self.num_examples_per_block * 2);
        let (out_queue_in, out_queue_out) = chan::sync(self.num_examples_per_block * 2);
        self.in_queue_in = Some(in_queue_in);
        self.out_queue_out = Some(out_queue_out);

        // read in examples
        let num_examples_per_block = self.num_examples_per_block.clone();
        let disk_buffer = self.disk_buffer.clone();
        let out_queue_in_clone = out_queue_in.clone();
        spawn(move|| {
            clear_in_queues(
                num_examples_per_block,
                in_queue_out,
                slot_in,
                out_queue_in_clone,
                disk_buffer
            )
        });

        // send out examples
        let disk_buffer = self.disk_buffer.clone();
        spawn(move|| {
            fill_out_queues(
                slot_out,
                out_queue_in,
                disk_buffer
            )
        });
    }
}


fn clear_in_queues(num_examples_per_block: usize,
                   in_queue_out: Receiver<ExampleWithScore>,
                   slot_in: Sender<usize>,
                   out_queue: chan::Sender<ExampleWithScore>,
                   disk_buffer: Arc<RwLock<DiskBuffer>>) {
    let mut in_block = VecDeque::with_capacity(num_examples_per_block);
    while let Ok(example) = in_queue_out.recv() {
        in_block.push_back(example);
        while let Some(example) = in_block.pop_front() { 
            let example_clone = example.clone();
            chan_select! {
                default => {
                    in_block.push_front(example);
                    break;
                },
                out_queue.send(example_clone) => {},
            }
        }
        if in_block.len() >= num_examples_per_block {
            let serialized_block = serialize(&in_block).unwrap();
            let block_idx = unlock_write!(disk_buffer).write(&serialized_block);
            slot_in.send(block_idx);
            in_block.clear();
        }
    }
    error!("In Queue in stratum was closed.");
}


fn fill_out_queues(slot_out: Receiver<usize>,
                   out_queue: chan::Sender<ExampleWithScore>,
                   disk_buffer: Arc<RwLock<DiskBuffer>>) {
    while let Ok(block_index) = slot_out.recv() {
        let out_block: Block = {
            let block_data: Vec<u8> = unlock_write!(disk_buffer).read(block_index);
            deserialize(&block_data).expect(
                "Cannot deserialize block."
            )
        };
        for example in out_block {
            out_queue.send(example);
        }
    }
    error!("Slot Queue in stratum was closed.");
}


fn get_locked<T>(t: T) -> Arc<RwLock<T>> {
    Arc::new(RwLock::new(t))
}


fn get_clone<T: Clone>(t: &mut Option<T>) -> Option<T> {
    if let Some(ref mut val) = t {
        Some(val.clone())
    } else {
        None
    }
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::RwLock;

    use labeled_data::LabeledData;
    use commons::ExampleWithScore;
    use super::super::get_disk_buffer;
    use super::Stratum;
    use super::InQueueSender;
    use super::OutQueueReceiver;

    #[test]
    fn test_stratum_one_by_one() {
        let (in_queue, out_queue) = get_in_out_queues();

        for i in 0..100 {
            let t = get_example(vec![i, 2, 3]);
            in_queue.send(t.clone()).unwrap();
            let retrieve = out_queue.recv().unwrap();
            assert_eq!(retrieve, t);
        }
    }

    // #[test]
    fn test_stratum_seq() {
        let (in_queue, out_queue) = get_in_out_queues();
        let mut examples = vec![];
        for i in 0..100 {
            let t = get_example(vec![i, 2, 3]);
            in_queue.send(t.clone()).unwrap();
            examples.push(t);
        }
        let mut output = vec![];
        for i in 0..100 {
            let retrieve = out_queue.recv().unwrap();
            output.push(retrieve);
        }
        output.sort_by_key(|t| (t.0).get_features()[0]);
        for i in 0..100 {
            assert_eq!(output[i], examples[i]);
        }
    }

    fn get_example(features: Vec<u8>) -> ExampleWithScore {
        let label: u8 = 0;
        let example = LabeledData::new(features, label);
        (example, (1.0, 0))
    }

    fn get_in_out_queues() -> (InQueueSender, OutQueueReceiver) {
        let disk_buffer = get_disk_buffer("unit-test-stratified.bin", 3, 100, 10);
        let mut stratum = Stratum::new(10, Arc::new(RwLock::new(disk_buffer)));
        stratum.run();
        (stratum.get_in_queue(), stratum.get_out_queue())
    }
}