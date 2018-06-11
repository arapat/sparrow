use rand;
use rayon::prelude::*;

use std::sync::Arc;
use std::sync::RwLock;

use std::sync::mpsc;
use std::thread::spawn;
use chan;

use commons::Model;
use commons::ExampleWithScore;

use commons::get_sign;
use commons::get_weight;

pub fn run_sampler(num_threads: usize,
                   initial_grid_size: f32,
                   average_window_size: usize,
                   loaded_examples: chan::Receiver<ExampleWithScore>,
                   updated_examples: chan::Sender<ExampleWithScore>,
                   sampled_examples: mpsc::SyncSender<ExampleWithScore>,
                   next_model: mpsc::Receiver<Model>) {
    let model = Arc::new(RwLock::new(vec![]));

    {
        let model_clone = model.clone();
        spawn(move|| {
            update_model(model_clone, next_model);
        });
    }
    {
        for _ in 0..num_threads {
            let grid_size = initial_grid_size.clone();
            let window_size = average_window_size.clone();
            let loaded_examples = loaded_examples.clone();
            let updated_examples = updated_examples.clone();
            let sampled_examples = sampled_examples.clone();
            let model = model.clone();
            spawn(move|| {
                sample(grid_size, window_size, loaded_examples,
                       updated_examples, sampled_examples, model);
            });
        }
    }
}


fn update_model(model: Arc<RwLock<Model>>, next_model: mpsc::Receiver<Model>) {
    for new_model in next_model.iter() {
        if let Ok(mut model_w) = model.write() {
            *model_w = new_model;
        }
    }
}


fn sample(initial_grid_size: f32,
          window_size: usize,
          loaded_examples: chan::Receiver<ExampleWithScore>,
          updated_examples: chan::Sender<ExampleWithScore>,
          sampled_examples: mpsc::SyncSender<ExampleWithScore>,
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
            sampled_examples.send((data.clone(), (updated_score, model_size)));
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
    use std::thread::spawn;

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
        let (loaded_sender, model_sender,
             updated_receiver, sampled_receiver) = get_sampler(num_threads);
        let mut model1 = Tree::new(2);
        model1.split(0, 0, 0.0, 0.0, 0.0);
        let mut model2 = Tree::new(2);
        model2.split(0, 0, 0.0, 1.0, 1.0);

        model_sender.send(vec![model1.clone()]).unwrap();
        sleep(Duration::from_millis(500));
        let mut examples: Vec<ExampleWithScore> = vec![];
        for i in 0..100 {
            let t = get_example(vec![i, 1, 2], 0.0);
            loaded_sender.send(t.clone());
            examples.push(t);
        }
        let mut all_updated = vec![];
        let mut all_sampled = vec![];
        for i in 0..100 {
            all_updated.push(updated_receiver.recv().unwrap());
            all_sampled.push(sampled_receiver.recv().unwrap());
        }
        let mut examples_i = examples.iter();
        if num_threads > 1 {
            all_updated.sort_by_key(|t| (t.0).get_features().clone());
            all_sampled.sort_by_key(|t| (t.0).get_features().clone());
        }
        let mut updated_i = all_updated.iter();
        let mut sampled_i = all_sampled.iter();
        for i in 0..100 {
            let t = examples_i.next().unwrap();
            assert_eq!(*updated_i.next().unwrap(), ((t.0).clone(), (0.0, 1)));
            assert_eq!(*sampled_i.next().unwrap(), ((t.0).clone(), (0.0, 1)));
        }

        model_sender.send(vec![model1, model2]).unwrap();
        sleep(Duration::from_millis(500));
        let mut examples = vec![];
        for i in 0..100 {
            let t = get_example(vec![i, 1, 2], 0.0);
            loaded_sender.send(t.clone());
            examples.push(t);
        }
        let mut all_updated = vec![];
        for i in 0..100 {
            all_updated.push(updated_receiver.recv().unwrap());
        }
        let mut examples_i = examples.iter();
        if num_threads > 1 {
            all_updated.sort_by_key(|t| (t.0).get_features().clone());
        }
        let mut updated_i = all_updated.iter();
        for i in 0..100 {
            let t = examples_i.next().unwrap();
            loaded_sender.send(t.clone());
            assert_eq!(*updated_i.next().unwrap(), ((t.0).clone(), (1.0, 2)));
        }

        let mut all_updated = vec![];
        for i in 0..100 {
            all_updated.push(updated_receiver.recv().unwrap());
        }
        let mut examples_i = examples.iter();
        if num_threads > 1 {
            all_updated.sort_by_key(|t| (t.0).get_features().clone());
        }
        let mut updated_i = all_updated.iter();
        for i in 0..100 {
            let t = examples_i.next().unwrap();
            assert_eq!(*updated_i.next().unwrap(), ((t.0).clone(), (1.0, 2)));
        }
    }

    fn get_sampler(num_threads: usize)
            -> (chan::Sender<ExampleWithScore>, mpsc::SyncSender<Model>,
                chan::Receiver<ExampleWithScore>, mpsc::Receiver<ExampleWithScore>) {
        let (loaded_sender, loaded_receiver) = chan::sync(100);
        let (updated_sender, updated_receiver) = chan::sync(100);
        let (sampled_sender, sampled_receiver) = mpsc::sync_channel(500);
        let (model_sender, model_receiver) = mpsc::sync_channel(100);
        run_sampler(num_threads,
                    1.0,
                    10,
                    loaded_receiver,
                    updated_sender,
                    sampled_sender,
                    model_receiver);
        (loaded_sender, model_sender, updated_receiver, sampled_receiver)
    }

    fn get_example(features: Vec<u8>, score: f32) -> ExampleWithScore {
        let label: u8 = 0;
        let example = LabeledData::new(features, label);
        (example, (score, 0))
    }
}