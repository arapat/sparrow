use std::sync::Arc;
use std::sync::RwLock;
use std::sync::mpsc::Receiver;
use rand::Rng;

use std::thread::spawn;
use rand::thread_rng;

use commons::ExampleWithScore;
use commons::performance_monitor::PerformanceMonitor;


/// Start the gatherer.
///
/// Fill the alternate memory buffer of the buffer loader
///
/// * `gather_new_sample`: the channle that the stratified storage sends sampled examples
/// * `new_sample_buffer`: the reference to the alternate memory buffer of the buffer loader
/// * `new_sample_capacity`: the size of the memory buffer of the buffer loader
pub fn run_gatherer(
    gather_new_sample: Receiver<ExampleWithScore>,
    new_sample_buffer: Arc<RwLock<Option<Vec<ExampleWithScore>>>>,
    new_sample_capacity: usize,
) {
    spawn(move || {
        loop {
            debug!("start filling the alternate buffer");
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


#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread::sleep;

    use std::sync::Arc;
    use std::sync::RwLock;
    use std::time::Duration;

    use commons::ExampleWithScore;
    use labeled_data::LabeledData;
    use super::run_gatherer;

    #[test]
    fn test_sampler() {
        let (gather_sender, gather_receiver) = mpsc::sync_channel(10);
        let mem_buffer = Arc::new(RwLock::new(None));
        run_gatherer(gather_receiver, mem_buffer.clone(), 100);

        let mut examples: Vec<ExampleWithScore> = vec![];
        for i in 0..100 {
            let t = get_example(vec![i, 1, 2], 0.0);
            gather_sender.send(t.clone()).unwrap();
            examples.push(t);
        }
        sleep(Duration::from_millis(1000));  // wait for the gatherer releasing the new sample
        let mut all_sampled = {
            let mut mem_buffer = mem_buffer.write().unwrap();
            assert!(mem_buffer.is_some());  // will poison the lock if this fails
            mem_buffer.take().unwrap()
        };
        all_sampled.sort_by_key(|t| (t.0).feature[0]);
        for (input, output) in examples.iter().zip(all_sampled.iter()) {
            assert_eq!(*input, *output);
        }
    }

    fn get_example(features: Vec<u8>, score: f32) -> ExampleWithScore {
        let label: u8 = 0;
        let example = LabeledData::new(features, label);
        (example, (score, 0))
    }
}