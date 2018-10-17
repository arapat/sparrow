use std::sync::Arc;
use std::sync::RwLock;
use crossbeam_channel as channel;

use std::sync::mpsc;
use std::thread::spawn;

use commons::ExampleWithScore;
use commons::performance_monitor::PerformanceMonitor;
use super::Strata;
use super::SPEED_TEST;

use commons::get_weight;


pub struct Assigners {
    updated_examples_r: channel::Receiver<ExampleWithScore>,
    strata: Arc<RwLock<Strata>>,
    stats_update_s: mpsc::SyncSender<(i8, (i32, f64))>,
}


impl Assigners {
    pub fn new(
        updated_examples_r: channel::Receiver<ExampleWithScore>,
        strata: Arc<RwLock<Strata>>,
        stats_update_s: mpsc::SyncSender<(i8, (i32, f64))>,
    ) -> Assigners {
        Assigners {
            updated_examples_r: updated_examples_r,
            strata: strata,
            stats_update_s: stats_update_s,
        }
    }

    pub fn run(&mut self, num_threads: usize) {
        for _ in 0..num_threads {
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
                    sender.unwrap().send((example, (score, version))).unwrap();
                    stats_update_s.send((index, (1, weight as f64))).unwrap();
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
    use std::sync::mpsc;
    use std::thread::sleep;
    use crossbeam_channel as channel;

    use std::sync::Arc;
    use std::sync::RwLock;
    use std::time::Duration;
    use self::channel::Sender;

    use labeled_data::LabeledData;
    use commons::ExampleWithScore;
    use super::super::Strata;
    use super::Assigners;

    #[test]
    fn test_assigner_1_thread() {
        let filename = "unittest-assigners1.bin";
        let (stats_update_r, sender, mut assigners) = get_assigner(filename);
        assigners.run(1);
        for i in 0..1 {
            for k in 0..3 {
                let t = get_example(vec![0, i, k], (2.0f32).powi(k as i32));
                sender.send(t.clone());
            }
        }

        sleep(Duration::from_millis(500));
        let num_examples: i32 = stats_update_r.try_iter().map(|t| (t.1).0).sum();
        assert_eq!(num_examples, 3);
        remove_file(filename).unwrap();
    }

    #[test]
    fn test_assigner_10_thread() {
        let filename = "unittest-assigners10.bin";
        let (stats_update_r, sender, mut assigners) = get_assigner(filename);
        assigners.run(10);
        for i in 0..10 {
            for k in 0..3 {
                let t = get_example(vec![0, i, k], (2.0f32).powi(k as i32));
                sender.send(t.clone());
            }
        }

        sleep(Duration::from_millis(500));
        let num_examples: i32 = stats_update_r.try_iter().map(|t| (t.1).0).sum();
        assert_eq!(num_examples, 30);
        remove_file(filename).unwrap();
    }

    fn get_assigner(
        filename: &str
    ) -> (mpsc::Receiver<(i8, (i32, f64))>, Sender<ExampleWithScore>, Assigners) {
        let strata = Arc::new(RwLock::new(Strata::new(100, 3, 10, filename)));
        let (updated_examples_send, updated_examples_recv) = channel::bounded(10);
        let (stats_update_s, stats_update_r) = mpsc::sync_channel(10);
        (
            stats_update_r,
            updated_examples_send,
            Assigners::new(updated_examples_recv, strata, stats_update_s)
        )
    }

    fn get_example(features: Vec<u8>, weight: f32) -> ExampleWithScore {
        let label: u8 = 0;
        let example = LabeledData::new(features, label);
        let score = weight.ln();
        (example, (score, 0))
    }
}