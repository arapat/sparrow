use std::collections::HashMap;
use std::collections::hash_map::Iter;
use chan::Sender;
use chan::Receiver;

use super::ExampleWithScore;

use chan;


pub struct MPMCMap {
    queue_size: usize,
    senders: HashMap<i8, Sender<ExampleWithScore>>,
    receivers: HashMap<i8, Receiver<ExampleWithScore>>
}


impl MPMCMap {
    pub fn new(queue_size: usize) -> MPMCMap {
        MPMCMap {
            queue_size: queue_size,
            senders: HashMap::new(),
            receivers: HashMap::new()
        }
    }

    pub fn iter_senders(&self) -> Iter<i8, Sender<ExampleWithScore>> {
        self.senders.iter()
    }

    pub fn iter_receivers(&self) -> Iter<i8, Receiver<ExampleWithScore>> {
        self.receivers.iter()
    }

    pub fn get(&mut self, key: i8) -> (Sender<ExampleWithScore>, Receiver<ExampleWithScore>) {
        if !self.senders.contains_key(&key) {
            self.insert(key);
        }
        (self.senders.get(&key).unwrap().clone(), self.receivers.get(&key).unwrap().clone())
    }

    fn insert(&mut self, key: i8) -> (Sender<ExampleWithScore>, Receiver<ExampleWithScore>) {
        let (sender, receiver) = {
            if self.queue_size == 0 {
                chan::async()
            } else {
                chan::sync(self.queue_size)
            }
        };
        self.senders.insert(key, sender);
        self.receivers.insert(key, receiver);
        (self.senders.get(&key).unwrap().clone(), self.receivers.get(&key).unwrap().clone())
    }
}
