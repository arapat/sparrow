use std::sync::mpsc;
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

use commons::ExampleWithScore;
use super::Block;
use super::InQueueSender;
use super::OutQueueReceiver;
use super::disk_buffer::DiskBuffer;


pub struct Stratum {
    // num_examples_per_block: usize,
    // disk_buffer: Arc<RwLock<DiskBuffer>>,
    pub in_queue_s: InQueueSender,
    pub out_queue_r: OutQueueReceiver,
}


macro_rules! unlock_write {
    ($lock:expr) => ($lock.write().unwrap());
}


impl Stratum {
    pub fn new(num_examples_per_block: usize, disk_buffer: Arc<RwLock<DiskBuffer>>) -> Stratum {
        // memory buffer for incoming examples
        let (in_queue_s, in_queue_r) = mpsc::sync_channel(num_examples_per_block * 2);
        // disk slot for storing most examples
        let (slot_s, slot_r) = mpsc::channel();
        // memory buffer for outgoing examples
        let (out_queue_s, out_queue_r) = chan::sync(num_examples_per_block * 2);
        // write in examples
        {
            let num_examples_per_block = num_examples_per_block.clone();
            let disk_buffer = disk_buffer.clone();
            let out_queue_s = out_queue_s.clone();
            spawn(move || {
                clear_in_queues(
                    num_examples_per_block,
                    in_queue_r,
                    slot_s,
                    out_queue_s,
                    disk_buffer,
                )
            });
        }
        // read out examples
        {
            let disk_buffer = disk_buffer.clone();
            spawn(move || {
                fill_out_queues(
                    slot_r,
                    out_queue_s,
                    disk_buffer,
                )
            });
        }
        Stratum {
            // num_examples_per_block: num_examples_per_block,
            // disk_buffer: disk_buffer,
            in_queue_s: in_queue_s,
            out_queue_r: out_queue_r,
        }
    }
}


fn clear_in_queues(num_examples_per_block: usize,
                   in_queue_out: Receiver<ExampleWithScore>,
                   slot_in: Sender<usize>,
                   out_queue: chan::Sender<ExampleWithScore>,
                   disk_buffer: Arc<RwLock<DiskBuffer>>) {
    let mut in_block = VecDeque::with_capacity(num_examples_per_block);
    loop {
        if let Ok(example) = in_queue_out.recv_timeout(Duration::from_millis(500)) {
            in_block.push_back(example);
            if in_block.len() >= num_examples_per_block {
                let serialized_block = serialize(&in_block).unwrap();
                let block_idx = unlock_write!(disk_buffer).write(&serialized_block);
                slot_in.send(block_idx).unwrap();
                in_block.clear();
            }
        } else {
            // if the number of examples is less than what requires to form a block,
            // they would stay in `in_block` forever and never write to disk.
            // In that case, `out_queue` might stuck if it is asked for examples
            // since it only has direct access to the examples on disk.
            // To address this issue, we send the examples to `out_queue` from time
            // to time here if `out_queue` is not full.
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
        }
    }
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


#[cfg(test)]
mod tests {
    use std::fs::remove_file;
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
        let filename = "unittest-stratum1.bin";
        let (in_queue, out_queue) = get_in_out_queues(filename, 100);

        for i in 0..2 {
            let t = get_example(vec![i, 2, 3]);
            in_queue.send(t.clone()).unwrap();
            let retrieve = out_queue.recv().unwrap();
            assert_eq!(retrieve, t);
        }
        remove_file(filename).unwrap();
    }

    #[test]
    fn test_stratum_seq() {
        let filename = "unittest-stratum2.bin";
        let (in_queue, out_queue) = get_in_out_queues(filename, 100);
        let mut examples = vec![];
        for i in 0..100 {
            let t = get_example(vec![i, 2, 3]);
            in_queue.send(t.clone()).unwrap();
            examples.push(t);
        }
        let mut output = vec![];
        for _ in 0..100 {
            let retrieve = out_queue.recv().unwrap();
            output.push(retrieve);
        }
        output.sort_by_key(|t| (t.0).feature[0]);
        for i in 0..100 {
            assert_eq!(output[i], examples[i]);
        }
        remove_file(filename).unwrap();
    }

    fn get_example(feature: Vec<u8>) -> ExampleWithScore {
        let label: u8 = 0;
        let example = LabeledData::new(feature, label);
        (example, (1.0, 0))
    }

    fn get_in_out_queues(filename: &str, size: usize) -> (InQueueSender, OutQueueReceiver) {
        let disk_buffer = get_disk_buffer(filename, 3, size, 10);
        let stratum = Stratum::new(10, Arc::new(RwLock::new(disk_buffer)));
        (stratum.in_queue_s.clone(), stratum.out_queue_r.clone())
    }
}