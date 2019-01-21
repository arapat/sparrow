use std::thread::sleep;
use std::thread::spawn;
use crossbeam_channel;
use bincode::serialize;
use bincode::deserialize;
use commons::channel;
use commons::performance_monitor::PerformanceMonitor;
use commons::ExampleWithScore;

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
    pub fn new(
        index: i8,
        num_examples_per_block: usize,
        disk_buffer: Arc<RwLock<DiskBuffer>>,
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
            spawn(move || {
                loop {
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
                    }
                }
            });
        }

        // Reading out data to outside
        spawn(move || {
            let mut pm = PerformanceMonitor::new();
            let mut num_stealed = 0;
            let mut out_block_ptr = (vec![]).into_iter();
            pm.start();
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
                        // they would stay in `in_queue` forever and never write to disk.
                        // We read from `in_queue` directly in this case.
                        example = in_queue_r.recv();
                        num_stealed += 1;
                    }
                }
                if example.is_some() {
                    out_queue_s.send(example.unwrap());
                    pm.update(1);
                } else {
                    sleep(Duration::from_millis(100));
                }
                if pm.get_duration() >= 5.0 {
                    debug!("stratum-queries, {}, {}, {}", index, pm.get_counts(), num_stealed);
                    pm.reset();
                    pm.start();
                    num_stealed = 0;
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
            let t = get_example(vec![i as f32, 2.0, 3.0]);
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
            let t = get_example(vec![i as f32, 2.0, 3.0]);
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

    fn get_example(feature: Vec<f32>) -> ExampleWithScore {
        let label: i8 = -1;
        let example = LabeledData::new(feature, label);
        (example, (1.0, 0))
    }

    fn get_in_out_queues(filename: &str, size: usize) -> (InQueueSender, OutQueueReceiver) {
        let disk_buffer = get_disk_buffer(filename, 3, size, 10);
        let stratum = Stratum::new(0, 10, Arc::new(RwLock::new(disk_buffer)));
        (stratum.in_queue_s.clone(), stratum.out_queue_r.clone())
    }
}
