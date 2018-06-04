use rand::Rng;
use rand::thread_rng;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc;
use std::sync::mpsc::Sender;
use std::sync::mpsc::Receiver;
use std::thread::sleep_ms;
use threadpool::ThreadPool;

use commons::Model;

use buffer_loader::normal_loader::NormalLoader;
use super::BlockModel;
use super::ExampleWithScore;
use super::sample_cons::SampleCons;


type NextLoader = Arc<Mutex<Option<NormalLoader>>>;
type ModelMutex = Arc<Mutex<Option<Model>>>;


pub struct BlocksWorker {
    sample_size: usize,

    sample_from_queue: Receiver<BlockModel>,
    processed_queue: Sender<ExampleWithScore>,

    next_model: ModelMutex,
    next_loader: NextLoader
}


impl BlocksWorker {
    pub fn new(sample_size: usize,
               sample_from_queue: Receiver<BlockModel>,
               processed_queue: Sender<ExampleWithScore>) -> BlocksWorker {
        BlocksWorker {
            sample_size: sample_size,

            sample_from_queue: sample_from_queue,
            processed_queue: processed_queue,

            next_model: Arc::new(Mutex::new(None)),
            next_loader: Arc::new(Mutex::new(None))
        }
    }

    pub fn run(&self, num_threads: usize, initial_grid_size: f32) {
        let (sender, receiver) = mpsc::channel();
        let curr_model: ModelMutex = Arc::new(Mutex::new(None));
        spawn(move|| {
            loop {
                let trees = self.get_new_mode();
                if let Ok(ref mut _model) = self.model_mutex.lock() {
                    *curr_model = **_model;
                }
                let sample_cons: SampleCons::new(self.sample_size);
                for (example, (score, base_node)) in receiver.iter() {
                    if base_node == trees.len() && sample_cons.append_data(&example, score) {
                        // new sample set is full
                        break;
                    }
                }
            }
        });
        spawn(move|| {
            let pool = ThreadPool::new(num_threads);
            let mut grid_size = initial_grid_size;
            let mut next_grid_size = Arc::new(Mutex::new(grid_size));
            loop {
                let examples: Block = self.sample_from_queue.recv();
                let trees = (
                    if let Ok(ref mut _model) = self.model_mutex.lock() {
                        **_model
                    } else {
                        None
                    }
                ).unwrap()
                while pool.active_count() >= num_threads {
                    sleep_ms(1000);
                }
                pool.execute(move|| {
                    sample(examples, trees, grid_size,
                           self.processed_queue.clone(), send_sample.clone())
                    // TODO: update grid size
                });
            }
        });
    }

    fn get_new_model(&mut self) -> Model {
        let model: Option<Model> = None;
        while model.is_none() {
            if let Ok(ref mut _model) = self.model_mutex.lock() {
                model = *_model;
            }
            sleep_ms(2000);
        }
        model.unwrap()
    }
}


fn sample(examples: Vec<ExampleWithScore>, trees: Model, grid_size: f32,
            send_updated: Sender<ExampleWithScore>, send_sample: Sender<ExampleWithScore>) {
    let model_size = trees.len();
    let mut sum_weights = (rand::thread_rng().gen()) * grid_size;
    for (data, (score, base_node)) in examples {
        let mut updated_score = score;
        for tree in trees[base_node..model_size].iter() {
            updated_score += tree.get_leaf_prediction(data);
        }
        send_updated.send((data, (updated_score, model_size)));

        let w = get_weight(data, updated_score);
        let next_sum_weight = sum_weights + w;
        let num_copies =
            (next_sum_weight / grid_size) as usize - (sum_weights / grid_size) as usize;
        max_repeat = max(max_repeat, num_copies);
        (0..num_copies).for_each(|_| {
            send_sample.send((data, (updated_score, model_size)));
        });
        sum_weights = next_sum_weight - num_copies as f32 * grid_size;
    }
}

// other random stuff
{
    let mut rand_val: f64 = rand::random() * self.sum_of_weight;
    let chosen = 0;
    for key in self.weights.keys() {
        rand_val -= self.weights.get(key).unwrap();
        if is_zero(rand_val) || rand_val < 0 {
            chosen = key;
            break;
        }
    }
    // get the block
    let (block_list, block) = self.bins.get(chosen).unwrap();

    if block_list.len() > 0 {
        // load data from disk
        let block_idx = block_list.pop_front().unwrap();
        let bin_data = self.disk_buffer.read(block_idx);
        let (block_examples, block_weight) = deserialize(bin_data).unwrap();

        // update bins
        self.bins.insert(chosen, (block_list, block));
        // update the weights
        let strata_weight = self.weights.get(chosen).unwrap();
        self.weights.insert(chosen, strata_weight - block_weight);
        self.sum_of_weight -= block_weight;

        (block_examples, block_weight)
    } else {  // return what is in the buffer
        // update bins
        self.bins.insert(chosen, (block_list, (vec![], 0.0)));
        // update the weights
        let strata_weight = self.weights.get(chosen).unwrap();
        self.weights.insert(chosen, 0.0);
        self.sum_of_weight -= strata_weight;

        block
    }
}