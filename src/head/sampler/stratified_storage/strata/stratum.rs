use std::thread::sleep;
use std::thread::spawn;
use crossbeam_channel;
use crossbeam_channel::Sender;
use bincode::serialize;
use bincode::deserialize;
use commons::channel;
use commons::performance_monitor::PerformanceMonitor;
use commons::ExampleWithScore;

use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

use super::Block;
use super::ArcBlockIter;
use super::QueueSender;
use super::QueueReceiver;
use super::disk_buffer::DiskBuffer;


pub struct Stratum {
    // num_examples_per_block: usize,
    // disk_buffer: Arc<RwLock<DiskBuffer>>,
    pub in_queue_s: QueueSender,
    pub in_queue_r: QueueReceiver,
    pub out_queue_r: QueueReceiver,
    pub out_block: ArcBlockIter,
    pub slot_s: Sender<usize>,
}


impl Stratum {
    pub fn new(
        index: i8,
        num_examples_per_block: usize,
        disk_buffer: Arc<RwLock<DiskBuffer>>,
        sampler_state: Arc<RwLock<bool>>,
    ) -> Stratum {
        // memory buffer for incoming examples
        let (in_queue_s, in_queue_r) =
            channel::bounded(num_examples_per_block * 2, &format!("stratum-i({})", index));
        // disk slot for storing most examples
        // maintained in a channel to support managing the strata with multiple threads (TODO)
        let (slot_s, slot_r) = crossbeam_channel::unbounded();
        // memory buffer for outgoing examples
        let (out_queue_s, out_queue_r) =
            channel::bounded(num_examples_per_block * 2, &format!("stratum-o({})", index));

        // Pushing in data from outside
        {
            let in_queue_r = in_queue_r.clone();
            let disk_buffer = disk_buffer.clone();
            let sampler_state = sampler_state.clone();
            let slot_s = slot_s.clone();
            spawn(move || {
                let mut state = true;
                while state {
                    if in_queue_r.len() >= num_examples_per_block {
                        let in_block: Vec<ExampleWithScore> =
                            (0..num_examples_per_block).map(|_| in_queue_r.recv().unwrap())
                                                       .collect();
                        let serialized_block = serialize(&(*in_block)).unwrap();
                        let mut disk = disk_buffer.write().unwrap();
                        let slot_index = disk.write(&serialized_block);
                        drop(disk);

                        slot_s.send(slot_index);
                    } else {
                        sleep(Duration::from_millis(100));
                        state = {
                            let sampler_state = sampler_state.read().unwrap();
                            *sampler_state
                        };
                    }
                }
            });
        }

        // Reading out data to outside
        let out_block = Arc::new(RwLock::new((vec![]).into_iter()));
        {
            let in_queue_r = in_queue_r.clone();
            let slot_r = slot_r.clone();
            let out_block_ptr = out_block.clone();
            spawn(move || {
                let mut pm = PerformanceMonitor::new();
                let mut state = true;
                pm.start();
                while state {
                    let mut example = {
                        let mut ptr = out_block_ptr.write().unwrap();
                        ptr.next()
                    };
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
                            let mut new_ptr = out_block.into_iter();
                            example = new_ptr.next();
                            {
                                let mut ptr = out_block_ptr.write().unwrap();
                                *ptr = new_ptr;
                            }
                        } else {
                            // if the number of examples is less than what requires to form a block,
                            // they would stay in `in_queue` forever and never write to disk.
                            // We read from `in_queue` directly in this case.
                            example = in_queue_r.recv();
                        }
                    }
                    if example.is_none() {
                        // error!("Stratum failed to read example.");
                        continue;
                    }
                    let example = example.unwrap();
                    out_queue_s.send(example.clone());
                    pm.update(1);
                    state = {
                        let sampler_state = sampler_state.read().unwrap();
                        *sampler_state
                    };
                }
            });
        }

        Stratum {
            // num_examples_per_block: num_examples_per_block,
            // disk_buffer: disk_buffer,
            in_queue_s: in_queue_s,
            in_queue_r: in_queue_r,
            out_queue_r: out_queue_r,
            out_block: out_block,
            slot_s: slot_s,
        }
    }
}


pub fn reset_block_scores(block_data: &[u8]) -> Vec<u8> {
    let block: Block = deserialize(&block_data).expect("Cannot deserialize block.");
    let new_block: Block = block.into_iter().map(|(example, (_, _))| (example, (0.0, 0))).collect();
    serialize(&(*new_block)).unwrap()
}


#[cfg(test)]
mod tests {
    use std::fs::remove_file;
    use std::sync::Arc;
    use std::sync::RwLock;

    use commons::labeled_data::LabeledData;
    use commons::ExampleWithScore;
    use super::Stratum;
    use super::QueueSender;
    use super::QueueReceiver;
    use super::super::get_disk_buffer;
    use TFeature;

    #[test]
    fn test_stratum_one_by_one() {
        let filename = "unittest-stratum1.bin";
        let (in_queue, out_queue) = get_in_out_queues(filename, 100);

        for i in 0..2 {
            let t = get_example(vec![i as TFeature, 2, 3]);
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
            let t = get_example(vec![i as TFeature, 2, 3]);
            in_queue.send(t.clone());
            examples.push(t);
        }
        let mut output = vec![];
        for _ in 0..100 {
            let retrieve = out_queue.recv().unwrap();
            output.push(retrieve);
        }
        output.sort_by(|t1, t2| (t1.0).feature[0].partial_cmp(&(t2.0).feature[0]).unwrap());
        for i in 0..100 {
            assert_eq!(output[i], examples[i]);
        }
        remove_file(filename).unwrap();
    }

    fn get_example(features: Vec<TFeature>) -> ExampleWithScore {
        let label: i8 = -1;
        let example = LabeledData::new(features, label);
        (example, (1.0, 0))
    }

    fn get_in_out_queues(filename: &str, size: usize) -> (QueueSender, QueueReceiver) {
        let disk_buffer = get_disk_buffer(filename, 3, size, 10);
        let stratum = Stratum::new(
            0, 10, Arc::new(RwLock::new(disk_buffer)), Arc::new(RwLock::new(true)));
        (stratum.in_queue_s.clone(), stratum.out_queue_r.clone())
    }
}
