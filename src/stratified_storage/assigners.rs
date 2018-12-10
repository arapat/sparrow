use std::sync::Arc;
use std::sync::RwLock;

use std::thread::spawn;

use commons::channel::Sender;
use commons::channel::Receiver;
use commons::ExampleWithScore;
use commons::performance_monitor::PerformanceMonitor;
use super::Strata;
use super::SPEED_TEST;

use commons::get_weight;


pub struct Assigners {
    updated_examples_r: Receiver<ExampleWithScore>,
    strata: Arc<RwLock<Strata>>,
    stats_update_s: Sender<(i8, (i32, f64))>,
    num_threads: usize,
}


impl Assigners {
    pub fn new(
        updated_examples_r: Receiver<ExampleWithScore>,
        strata: Arc<RwLock<Strata>>,
        stats_update_s: Sender<(i8, (i32, f64))>,
        num_threads: usize,
    ) -> Assigners {
        Assigners {
            updated_examples_r: updated_examples_r,
            strata: strata,
            stats_update_s: stats_update_s,
            num_threads: num_threads,
        }
    }

    pub fn run(&self) {
        for _ in 0..self.num_threads {
            let updated_examples_r = self.updated_examples_r.clone();
            let strata = self.strata.clone();
            let stats_update_s = self.stats_update_s.clone();
            spawn(move|| {
                let mut rotate = 0;
                let mut pm = PerformanceMonitor::new();
                pm.start();
                while let Some(ret) = updated_examples_r.recv() {
                    let (example, (score, version)) = ret;
                    let weight = get_weight(&example, score);
                    let index = {
                        if SPEED_TEST {
                            rotate = (rotate + 1) % 10;
                            rotate + 1
                        } else {
                            weight.log2() as i8
                        }
                    };
                    let read_strata = strata.read().unwrap();
                    let mut sender = read_strata.get_in_queue(index);
                    drop(read_strata);
                    if sender.is_none() {
                        let mut strata = strata.write().unwrap();
                        sender = Some(strata.create(index).0);
                    }
                    sender.unwrap().send((example, (score, version)));
                    stats_update_s.send((index, (1, weight as f64)));
                    pm.update(1);
                    pm.write_log("selector");
                }
            });
        }
    }
}


#[cfg(test)]
mod tests {
    use std::fs::remove_file;
    use std::thread::sleep;
    use commons::channel;

    use std::sync::Arc;
    use std::sync::RwLock;
    use std::time::Duration;
    use self::channel::Receiver;
    use self::channel::Sender;

    use labeled_data::LabeledData;
    use commons::ExampleWithScore;
    use super::super::Strata;
    use super::Assigners;

    #[test]
    fn test_assigner_1_thread() {
        let filename = "unittest-assigners1.bin";
        let (stats_update_r, sender, assigners) = get_assigner(filename, 1);
        assigners.run();
        for i in 0..1 {
            for k in 0..3 {
                let t = get_example(vec![0.0, i as f32, k as f32], (2.0f32).powi(k as i32));
                sender.send(t.clone());
            }
        }

        sleep(Duration::from_millis(500));
        let mut num_examples = 0;
        while let Some(t) = stats_update_r.try_recv() {
            num_examples += (t.1).0;
        }
        assert_eq!(num_examples, 3);
        remove_file(filename).unwrap();
    }

    #[test]
    fn test_assigner_10_thread() {
        let filename = "unittest-assigners10.bin";
        let (stats_update_r, sender, assigners) = get_assigner(filename, 10);
        assigners.run();
        for i in 0..10 {
            for k in 0..3 {
                let t = get_example(vec![0.0, i as f32, k as f32], (2.0f32).powi(k as i32));
                sender.send(t.clone());
            }
        }

        sleep(Duration::from_millis(500));
        let mut num_examples = 0;
        while let Some(t) = stats_update_r.try_recv() {
            num_examples += (t.1).0;
        }
        assert_eq!(num_examples, 30);
        remove_file(filename).unwrap();
    }

    fn get_assigner(
        filename: &str,
        num_threads: usize,
    ) -> (Receiver<(i8, (i32, f64))>, Sender<ExampleWithScore>, Assigners) {
        let strata = Arc::new(RwLock::new(Strata::new(100, 3, 10, filename)));
        let (updated_examples_send, updated_examples_recv) = channel::bounded(10, "updated-examples");
        let (stats_update_s, stats_update_r) = channel::bounded(100, "stats");
        (
            stats_update_r,
            updated_examples_send,
            Assigners::new(updated_examples_recv, strata, stats_update_s, num_threads)
        )
    }

    fn get_example(features: Vec<f32>, weight: f32) -> ExampleWithScore {
        let label: i8 = -1;
        let example = LabeledData::new(features, label);
        let score = weight.ln();
        (example, (score, 0))
    }
}
