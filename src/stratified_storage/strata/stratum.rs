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

        // read in examples
        let (in_queue_in, in_queue_out) = mpsc::sync_channel(self.num_examples_per_block * 2);
        self.in_queue_in = Some(in_queue_in);

        let num_examples_per_block = self.num_examples_per_block.clone();
        let disk_buffer = self.disk_buffer.clone();
        spawn(move|| {
            clear_in_queues(
                num_examples_per_block,
                in_queue_out,
                slot_in,
                disk_buffer
            )
        });

        // send out examples
        let (out_queue_in, out_queue_out) = chan::sync(num_examples_per_block * 2);
        self.out_queue_out = Some(out_queue_out);

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
                   disk_buffer: Arc<RwLock<DiskBuffer>>) {
    let mut in_block = Vec::with_capacity(num_examples_per_block);
    loop {
        let example_with_score = in_queue_out.recv().unwrap();
        in_block.push(example_with_score);
        if in_block.len() >= num_examples_per_block {
            let serialized_block = serialize(&in_block).unwrap();
            let block_idx = unlock_write!(disk_buffer).write(&serialized_block);
            slot_in.send(block_idx);
            in_block.clear();
        }
    }
}


fn fill_out_queues(slot_out: Receiver<usize>,
                   out_queue: chan::Sender<ExampleWithScore>,
                   disk_buffer: Arc<RwLock<DiskBuffer>>) {
    loop {
        let block_index = slot_out.recv().unwrap();
        let block_data: Vec<u8> = unlock_write!(disk_buffer).read(block_index);
        let out_block: Vec<ExampleWithScore> = deserialize(&block_data).unwrap();
        for example in out_block {
            out_queue.send(example);
        }
    }
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