mod bitmap;
mod disk_buffer;

use std::thread::sleep;
use std::thread::spawn;
use bincode::serialize;
use bincode::deserialize;

use std::collections::HashMap;
use std::collections::vec_deque::VecDeque;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

use commons::Example;
use commons::TLabel;
use labeled_data::LabeledData;

use super::ExampleWithScore;
use super::mpmc_map::MPMCMap;
use self::bitmap::BitMap;
use self::disk_buffer::DiskBuffer;


type Block = Vec<ExampleWithScore>;
type LockedBlock = Arc<RwLock<VecDeque<ExampleWithScore>>>;
type LockedIndices = Arc<RwLock<VecDeque<usize>>>;
type HashMapExamples = HashMap<i8, LockedBlock>;
type HashMapIndices = HashMap<i8, LockedIndices>;


pub struct Strata {
    num_examples_per_block: usize,

    disk_buffer: Arc<RwLock<DiskBuffer>>,
    bitmap: Arc<RwLock<BitMap>>,

    in_queue: Arc<RwLock<MPMCMap>>,
    in_block: Arc<RwLock<HashMapExamples>>,
    slot_indices: Arc<RwLock<HashMapIndices>>,
    out_block: Arc<RwLock<HashMapExamples>>,
    out_queue: Arc<RwLock<MPMCMap>>
}


macro_rules! unlock_read {
    ($lock:expr) => ($lock.read().unwrap());
}


macro_rules! unlock_write {
    ($lock:expr) => ($lock.write().unwrap());
}


impl Strata {
    pub fn new(num_examples: usize,
               num_examples_per_block: usize,
               in_queue: Arc<RwLock<MPMCMap>>,
               out_queue: Arc<RwLock<MPMCMap>>) -> Strata {
        let block_size = get_block_size(num_examples_per_block);
        let num_disk_block = (num_examples + num_examples_per_block - 1) / num_examples_per_block;
        let buffer = DiskBuffer::new("stratified_buffer.bin", block_size, num_disk_block);
        let disk_buffer = get_locked(buffer);
        let bitmap = get_locked(BitMap::new(num_disk_block, false));
        let in_block = get_locked(HashMap::new());
        let slot_indices = get_locked(HashMap::new());
        let out_block = get_locked(HashMap::new());
        Strata {
            num_examples_per_block: num_examples_per_block,

            disk_buffer: disk_buffer,
            bitmap: bitmap,

            in_queue: in_queue,
            in_block: in_block,
            slot_indices: slot_indices,
            out_block: out_block,
            out_queue: out_queue
        }
    }

    pub fn run(&mut self, num_in_threads: usize, num_out_threads: usize) {
        for _ in 0..num_in_threads {
            let num_examples_per_block = self.num_examples_per_block.clone();
            let in_queue = self.in_queue.clone();
            let in_block = self.in_block.clone();
            let slot_indices = self.slot_indices.clone();
            let bitmap = self.bitmap.clone();
            let disk_buffer = self.disk_buffer.clone();
            spawn(move|| {
                clear_in_queues(
                    num_examples_per_block,
                    in_queue,
                    in_block,
                    slot_indices,
                    bitmap,
                    disk_buffer
                )
            });
        }

        for _ in 0..num_out_threads {
            let num_examples_per_block = self.num_examples_per_block.clone();
            let slot_indices = self.slot_indices.clone();
            let out_block = self.out_block.clone();
            let out_queue = self.out_queue.clone();
            let bitmap = self.bitmap.clone();
            let disk_buffer = self.disk_buffer.clone();
            spawn(move|| {
                fill_out_queues(
                    num_examples_per_block,
                    slot_indices,
                    out_block,
                    out_queue,
                    bitmap,
                    disk_buffer
                )
            });
        }
    }
}


fn clear_in_queues(num_examples_per_block: usize,
                   in_queue: Arc<RwLock<MPMCMap>>,
                   in_block: Arc<RwLock<HashMapExamples>>,
                   slot_indices: Arc<RwLock<HashMapIndices>>,
                   bitmap: Arc<RwLock<BitMap>>,
                   disk_buffer: Arc<RwLock<DiskBuffer>>) {
    loop {
        let ret = {
            let mut ret: Option<(i8, ExampleWithScore)> = None;
            for (key, receiver) in unlock_read!(in_queue).iter_receivers() {
                chan_select! {
                    default => {},
                    receiver.recv() -> val => {
                        ret = Some((*key, val.unwrap()));
                        break;
                    }
                }
            }
            ret
        };
        if let Some((index, example_with_score)) = ret {
            let locked_block = get_deque(index, &in_block, num_examples_per_block);
            let write_back_block: Option<Block> = {
                let mut block = unlock_write!(locked_block);
                block.push_back(example_with_score);
                if block.len() >= num_examples_per_block {
                    let ret = block.clone();
                    block.clear();
                    Some(Vec::from(ret))
                } else {
                    None
                }
            };
            if let Some(block) = write_back_block {
                let block_idx = {
                    let mut bitmap_w = unlock_write!(bitmap);
                    let idx = bitmap_w.get_first_free().expect(
                        "No free slot available."
                    );
                    bitmap_w.mark_filled(idx);
                    idx
                };
                let serialized_block = serialize(&block).unwrap();
                let locked_slot = get_deque(index, &slot_indices, num_examples_per_block);
                unlock_write!(disk_buffer).write(block_idx, &serialized_block);
                unlock_write!(locked_slot).push_back(block_idx);
            }
        } else {
            sleep(Duration::from_millis(500));
        }
    }
}


fn fill_out_queues(num_examples_per_block: usize,
                   slot_indices: Arc<RwLock<HashMapIndices>>,
                   out_block: Arc<RwLock<HashMapExamples>>,
                   out_queue: Arc<RwLock<MPMCMap>>,
                   bitmap: Arc<RwLock<BitMap>>,
                   disk_buffer: Arc<RwLock<DiskBuffer>>) {
    loop {
        {
            for (key, locked_slot) in unlock_read!(slot_indices).iter() {
                let mut slot_w = unlock_write!(locked_slot);
                let block: LockedBlock = get_deque(*key, &out_block, num_examples_per_block);
                if slot_w.len() > 0 && unlock_read!(block).len() == 0 {
                    let idx = slot_w.pop_front().unwrap();
                    unlock_write!(bitmap).mark_free(idx);

                    let block_data: Vec<u8> = unlock_write!(disk_buffer).read(idx);
                    let block_examples: Block = deserialize(&block_data).unwrap();
                    let locked_block = get_locked(VecDeque::from(block_examples));
                    unlock_write!(out_block).insert(*key, locked_block);
                }
            }
        }
        {
            for (index, locked_block) in unlock_read!(out_block).iter() {
                let mut block_w = unlock_write!(locked_block);
                let (out_sender, _) = unlock_write!(out_queue).get(*index);
                let mut leftover = None;
                while let Some(example) = block_w.pop_front() {
                    chan_select! {
                        default => {
                            leftover = Some(example.clone());
                            break;
                        },
                        out_sender.send(example.clone()) => {}
                    }
                }
                if leftover.is_some() {
                    block_w.push_front(leftover.unwrap());
                }
            }
        }
    }
}


fn get_block_size(num_examples_per_block: usize) -> usize {
    let example: Example = LabeledData::new(vec![], 0 as TLabel);
    let example_with_score: ExampleWithScore = (example, (0.0, 0));
    let block: Block = vec![example_with_score; num_examples_per_block];
    let serialized_block: Vec<u8> = serialize(&block).unwrap();
    serialized_block.len()
}


fn get_deque<T: Clone>(key: i8,
                       deques: &Arc<RwLock<HashMap<i8, Arc<RwLock<VecDeque<T>>>>>>,
                       queue_size: usize) -> Arc<RwLock<VecDeque<T>>> {
    unlock_write!(deques).entry(key).or_insert(
        Arc::new(RwLock::new(VecDeque::with_capacity(queue_size)))
    ).clone()
}


fn get_locked<T>(t: T) -> Arc<RwLock<T>> {
    Arc::new(RwLock::new(t))
}