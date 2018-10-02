use rand;
use rand::Rng;
use rand::thread_rng;
use rayon::prelude::*;

use std::sync::Arc;
use std::sync::RwLock;

use std::sync::mpsc;
use std::sync::mpsc::channel;
use std::thread::spawn;
use chan;

use commons::ExampleWithScore;
use commons::Model;
use commons::performance_monitor::PerformanceMonitor;

use commons::get_sign;
use commons::get_weight;


/// Start the sampler.
///
/// Sampler read examples loaded from the stratified storage, and do
/// the following two tasks on these examples:
///
/// 1. update their scores,
/// 2. sample and send the sampled examples to the `BufferLoader` object.
///
/// The function uses the minimum variance sampling method to sample from
/// the examples loaded from the stratified storage.
/// The initial grid size is specified as the parameter to the function.
/// Over the sampling process, the grid size is updated as the moving average
/// of the weights of the `average_window_size` number of examples that
/// scanned by the sampler.
///
/// * `num_thread`: the number of threads that runs the sampler
/// * `initial_grid_size`: the initial gap between the two examples sampled using
/// the minimum vairance sampling method.
/// * `average_window_size`: the window size used in computing the moving average of
/// the grid size.
/// * `loaded_examples`: the channle that the stratified storage sends examples loaded from disk.
/// * `updated_examples`: the channel that the sampler will send the examples after
/// updating their scores.
/// * `new_sample_buffer`: the reference to the alternate memory buffer of the buffer loader
/// * `new_sample_capacity`: the size of the memory buffer of the buffer loader
pub fn run_sampler(
    num_threads: usize,
    initial_grid_size: f32,
    average_window_size: usize,
    loaded_examples: chan::Receiver<ExampleWithScore>,
    updated_examples: chan::Sender<ExampleWithScore>,
    next_model: mpsc::Receiver<Model>,
    new_sample_buffer: Arc<RwLock<Option<Vec<ExampleWithScore>>>>,
    new_sample_capacity: usize,
) {
    let model = Arc::new(RwLock::new(vec![]));
    // TODO: fixed the capacity of the new sample channel
    let (gen_new_sample, gather_new_sample) = channel();
    // Update the latest model used in the sampler
    {
        let model = model.clone();
        spawn(move || {
            for new_model in next_model.iter() {
                let model_len = new_model.len();
                let model = model.write();
                *(model.unwrap()) = new_model;
                info!("sampler model update, {}", model_len);
            }
        });
    }
    // Activate all sampler
    {
        for _ in 0..num_threads {
            let grid_size = initial_grid_size.clone();
            let window_size = average_window_size.clone();
            let loaded_examples = loaded_examples.clone();
            let updated_examples = updated_examples.clone();
            let gen_new_sample = gen_new_sample.clone();
            let model = model.clone();
            spawn(move || {
                sample(grid_size, window_size, loaded_examples,
                       updated_examples, gen_new_sample, model);
            });
        }
    }
    // Activate sample gatherer
    {
        spawn(move || {
            loop {
                debug!("start generating new sample");
                let mut pm = PerformanceMonitor::new();
                pm.start();

                let mut new_sample = Vec::with_capacity(new_sample_capacity);
                while new_sample.len() < new_sample_capacity {
                    if let Ok(t) = gather_new_sample.recv() {
                        new_sample.push(t);
                    }
                }
                thread_rng().shuffle(&mut new_sample);
                {
                    let new_sample_lock = new_sample_buffer.write();
                    *(new_sample_lock.unwrap()) = Some(new_sample);
                }
                debug!("new-sample, {}", new_sample_capacity as f32 / pm.get_duration());
            }
        });
    }
}


fn sample(initial_grid_size: f32,
          window_size: usize,
          loaded_examples: chan::Receiver<ExampleWithScore>,
          updated_examples: chan::Sender<ExampleWithScore>,
          sampled_examples: mpsc::Sender<ExampleWithScore>,
          model: Arc<RwLock<Model>>) {
    let mut grid_size = initial_grid_size;
    let mut grid = -rand::random::<f32>() * grid_size;
    let rou = 1.0 / window_size as f32;
    while let Some((data, (score, base_node))) = loaded_examples.recv() {
        let (updated_score, model_size) = {
            let trees = model.read().unwrap();
            let model_size = trees.len();
            let inc_score: f32 = {
                trees[base_node..model_size].par_iter().map(|tree| {
                    tree.get_leaf_prediction(&data)
                }).sum()
            };
            (score + inc_score, model_size)
        };

        updated_examples.send((data.clone(), (updated_score, model_size)));
        let w = get_weight(&data, updated_score);
        grid_size = grid_size * (1.0 - rou) + w * rou;
        grid += w;
        while get_sign(grid as f64) >= 0 {
            sampled_examples.send((data.clone(), (updated_score, model_size))).unwrap();
            grid -= grid_size;
        }
    }
    error!("Loaded example queue is closed.");
}


#[cfg(test)]
mod tests {
    use chan;
    use std::sync::mpsc;
    use std::thread::sleep;

    use std::sync::Arc;
    use std::sync::RwLock;
    use std::time::Duration;

    use commons::ExampleWithScore;
    use commons::Model;
    use labeled_data::LabeledData;
    use tree::Tree;
    use super::run_sampler;


    #[test]
    fn test_sampler_1_thread() {
        test_sampler(1);
    }

    #[test]
    fn test_sampler_10_threads() {
        test_sampler(10);
    }

    fn test_sampler(num_threads: usize) {
        let (loaded_sender, model_sender, updated_receiver, mem_buffer) = get_sampler(num_threads);
        let mut model1 = Tree::new(2);
        model1.split(0, 0, 0.0, 0.0, 0.0);
        let mut model2 = Tree::new(2);
        model2.split(0, 0, 0.0, 1.0, 1.0);

        model_sender.send(vec![model1.clone()]).unwrap();
        sleep(Duration::from_millis(1000));  // wait for the sampler receiving the new model
        let mut examples: Vec<ExampleWithScore> = vec![];
        for i in 0..100 {
            let t = get_example(vec![i, 1, 2], 0.0);
            loaded_sender.send(t.clone());
            examples.push(t);
        }
        let mut all_updated: Vec<ExampleWithScore> = vec![];
        for _ in 0..100 {
            all_updated.push(updated_receiver.recv().unwrap());
        }
        sleep(Duration::from_millis(1000));  // wait for the gatherer releasing the new sample
        let mut all_sampled = {
            let mut mem_buffer = mem_buffer.write().unwrap();
            assert!(mem_buffer.is_some());  // will poison the lock if this fails
            mem_buffer.take().unwrap()
        };
        let mut examples_i = examples.iter();
        all_updated.sort_by_key(|t| (t.0).get_features().clone());
        all_sampled.sort_by_key(|t| (t.0).get_features().clone());
        let mut updated_i = all_updated.iter();
        let mut sampled_i = all_sampled.iter();
        for _ in 0..100 {
            let t = examples_i.next().unwrap();
            assert_eq!(*updated_i.next().unwrap(), ((t.0).clone(), (0.0, 1)));
            assert_eq!(*sampled_i.next().unwrap(), ((t.0).clone(), (0.0, 1)));
        }

        model_sender.send(vec![model1, model2]).unwrap();
        sleep(Duration::from_millis(1000));  // wait for the sampler receiving the new model
        let mut examples = vec![];
        for i in 0..100 {
            let t = get_example(vec![i, 1, 2], 0.0);
            loaded_sender.send(t.clone());
            examples.push(t);
        }
        let mut all_updated = vec![];
        for _ in 0..100 {
            all_updated.push(updated_receiver.recv().unwrap());
        }
        let mut examples_i = examples.iter();
        if num_threads > 1 {
            all_updated.sort_by_key(|t| (t.0).get_features().clone());
        }
        let mut updated_i = all_updated.iter();
        for _ in 0..100 {
            let t = examples_i.next().unwrap();
            loaded_sender.send(t.clone());
            assert_eq!(*updated_i.next().unwrap(), ((t.0).clone(), (1.0, 2)));
        }

        let mut all_updated = vec![];
        for _ in 0..100 {
            all_updated.push(updated_receiver.recv().unwrap());
        }
        let mut examples_i = examples.iter();
        if num_threads > 1 {
            all_updated.sort_by_key(|t| (t.0).get_features().clone());
        }
        let mut updated_i = all_updated.iter();
        for _ in 0..100 {
            let t = examples_i.next().unwrap();
            assert_eq!(*updated_i.next().unwrap(), ((t.0).clone(), (1.0, 2)));
        }
    }

    fn get_sampler(num_threads: usize) -> (
        chan::Sender<ExampleWithScore>,
        mpsc::SyncSender<Model>,
        chan::Receiver<ExampleWithScore>,
        Arc<RwLock<Option<Vec<ExampleWithScore>>>>,
    ) {
        let (loaded_sender, loaded_receiver) = chan::sync(100);
        let (updated_sender, updated_receiver) = chan::sync(100);
        let (model_sender, model_receiver) = mpsc::sync_channel(100);
        let mem_buffer = Arc::new(RwLock::new(None));
        run_sampler(num_threads,
                    1.0,
                    10,
                    loaded_receiver,
                    updated_sender,
                    model_receiver,
                    mem_buffer.clone(),
                    100);
        (loaded_sender, model_sender, updated_receiver, mem_buffer)
    }

    fn get_example(features: Vec<u8>, score: f32) -> ExampleWithScore {
        let label: u8 = 0;
        let example = LabeledData::new(features, label);
        (example, (score, 0))
    }
}