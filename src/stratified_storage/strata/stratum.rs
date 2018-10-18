use std::thread::sleep;
use std::thread::spawn;
use bincode::serialize;
use bincode::deserialize;
use crossbeam_channel as channel;

use std::collections::vec_deque::VecDeque;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

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


impl Stratum {
    pub fn new(num_examples_per_block: usize, disk_buffer: Arc<RwLock<DiskBuffer>>) -> Stratum {
        // memory buffer for incoming examples
        let (in_queue_s, in_queue_r) = channel::bounded(num_examples_per_block * 2);
        // disk slot for storing most examples
        // maintained in a channel to support managing the strata with multiple threads (TODO)
        let (slot_s, slot_r) = channel::unbounded();
        // memory buffer for outgoing examples
        let (out_queue_s, out_queue_r) = channel::bounded(num_examples_per_block * 2);

        let in_block = Arc::new(RwLock::new(VecDeque::with_capacity(num_examples_per_block)));
        // Pushing in data from outside
        {
            let in_block = in_block.clone();
            let disk_buffer = disk_buffer.clone();
            spawn(move || {
                while let Some(example) = in_queue_r.recv() {
                    let mut in_block = in_block.write().unwrap();
                    in_block.push_back(example);
                    if in_block.len() >= num_examples_per_block {
                        let serialized_block = serialize(&(*in_block)).unwrap();
                        let mut disk = disk_buffer.write().unwrap();
                        let slot_index = disk.write(&serialized_block);
                        drop(disk);

                        slot_s.send(slot_index);
                        in_block.clear();
                    }
                    drop(in_block);
                }
            });
        }

        // Reading out data to outside
        spawn(move || {
            let mut out_block_ptr = (vec![]).into_iter();
            loop {
                let mut example = out_block_ptr.next();
                if example.is_none() {
                    if let Some(block_index) = slot_r.try_recv() {
                        let out_block: Block = {
                            let mut disk = disk_buffer.write().unwrap();
                            let block_data: Vec<u8> = disk.read(block_index);
                            drop(disk);
                            deserialize(&block_data).expect(
                                "Cannot deserialize block."
                            )
                        };
                        out_block_ptr = out_block.into_iter();
                        example = out_block_ptr.next();
                    } else {
                        // if the number of examples is less than what requires to form a block,
                        // they would stay in `in_block` forever and never write to disk.
                        // We read from `in_block` directly in this case.
                        let mut in_block = in_block.write().unwrap();
                        example = in_block.pop_front();
                        drop(in_block);
                    }
                }
                if example.is_some() {
                    out_queue_s.send(example.unwrap());
                } else {
                    sleep(Duration::from_millis(1000));
                }
            }
        });

        Stratum {
            // num_examples_per_block: num_examples_per_block,
            // disk_buffer: disk_buffer,
            in_queue_s: in_queue_s,
            out_queue_r: out_queue_r,
        }
    }
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
            in_queue.send(t.clone());
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
            in_queue.send(t.clone());
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