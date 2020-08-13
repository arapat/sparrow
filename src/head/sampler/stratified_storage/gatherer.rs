use rand::Rng;

use std::fs::rename;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::mpsc::Sender;
use std::thread::spawn;
use rand::thread_rng;
use bincode::serialize;

use SampleMode;
use commons::channel::Receiver;
use commons::Model;
use commons::io::write_all;
use commons::ExampleWithScore;
use commons::packet::TaskPacket;
use commons::performance_monitor::PerformanceMonitor;
use commons::persistent_io::write_sample_local;
use commons::persistent_io::write_sample_s3;


pub struct Gatherer {
    gather_new_sample:      Receiver<((ExampleWithScore, u32), u32)>,
    new_sample_capacity:    usize,
    model:                  Arc<RwLock<Model>>,
    pub counter:            Arc<RwLock<Vec<u32>>>,
    exp_name:               String,
}


impl Gatherer {
    /// * `new_sample_capacity`: the size of the memory buffer of the buffer loader
    pub fn new(
        gather_new_sample:      Receiver<((ExampleWithScore, u32), u32)>,
        new_sample_capacity:    usize,
        model:                  Arc<RwLock<Model>>,
        exp_name:               String,
    ) -> Gatherer {
        Gatherer {
            gather_new_sample:   gather_new_sample,
            new_sample_capacity: new_sample_capacity,
            model:               model,
            counter:             Arc::new(RwLock::new(vec![])),
            exp_name:            exp_name,
        }
    }

    /// Start the gatherer.
    ///
    /// Fill the alternate memory buffer of the buffer loader
    pub fn run(&self, mode: SampleMode, packet_sender: Sender<(Option<String>, TaskPacket)>) {
        let new_sample_capacity = self.new_sample_capacity;
        let gather_new_sample = self.gather_new_sample.clone();
        let model = self.model.clone();
        let exp_name = self.exp_name.clone();
        info!("Starting non-blocking gatherer");
        spawn(move || {
            let mut version = 0;
            loop {
                version += 1;
                let write_sample_func = match mode {
                    SampleMode::LOCAL => {
                        write_sample_local
                    },
                    SampleMode::S3 => {
                        write_sample_s3
                    },
                };
                gather(
                    new_sample_capacity,
                    gather_new_sample.clone(),
                    write_sample_func,
                    version,
                    model.clone(),
                    exp_name.as_str(),
                );

                let mut packet = TaskPacket::new();
                packet.set_sample_version(version);
                packet_sender.send((None, packet)).unwrap();
            }
        });
    }
}


fn gather<F>(
    new_sample_capacity: usize,
    gather_new_sample: Receiver<((ExampleWithScore, u32), u32)>,
    broadcast_handler: F,
    version: usize,
    model: Arc<RwLock<Model>>,
    exp_name: &str,
) where F: Fn(Vec<ExampleWithScore>, Model, usize, &str) {
    debug!("sampler, start, generate sample");
    let mut pm = PerformanceMonitor::new();
    pm.start();

    let mut new_sample: Vec<ExampleWithScore> = Vec::with_capacity(new_sample_capacity);
    let mut total_scanned = 0;
    let mut num_unique = 0;
    let mut num_unique_positive = 0;
    let mut num_total_positive = 0;
    let mut last_log_count = 0;
    while new_sample.len() < new_sample_capacity {
        if let Some(((example, mut c), num_scanned)) = gather_new_sample.recv() {
            // `c` is the number of times this example should be put into the sample set
            if example.0.label > 0 {
                num_unique_positive += 1;
                num_total_positive += c;
            }
            num_unique += 1;
            while new_sample.len() < new_sample_capacity && c > 0 {
                new_sample.push(example.clone());
                c -= 1;
            }
            total_scanned += num_scanned;
        }
        if new_sample.len() - last_log_count >= new_sample_capacity / 10 {
            debug!("sampler, progress, {}", new_sample.len());
            last_log_count = new_sample.len();
        }
    }
    thread_rng().shuffle(&mut new_sample);
    // TODO: count number of examples fall, make sure the numbering is the same as the assignments
    let model = {
        let lock = model.read().unwrap();
        lock.clone()
    };
    debug!("sampler, finished, generate sample, {}, {}, {}, {}, {}, {}",
           total_scanned, new_sample.len(), num_total_positive, num_unique, num_unique_positive,
           model.size());
    // Create a snapshot for continous training
    let filename = "latest_sample.bin".to_string() + "_WRITING";
    write_all(&filename, &serialize(&(version, new_sample.clone(), &model)).unwrap())
        .expect("Failed to write the sample set to file for snapshot");
    rename(filename, "latest_sample.bin".to_string()).unwrap();
    // Send the sample to the broadcast handler
    broadcast_handler(new_sample, model, version, exp_name);
    let duration = pm.get_duration();
    debug!("sample-gatherer, {}, {}", duration, new_sample_capacity as f32 / duration);
}


/*
#[cfg(test)]
mod tests {
    use std::thread::sleep;

    use std::sync::Arc;
    use std::sync::RwLock;
    use std::time::Duration;

    use commons::channel;
    use commons::ExampleWithScore;
    use commons::Model;
    use commons::labeled_data::LabeledData;
    use commons::persistent_io::load_sample_local;
    use super::Gatherer;
    use super::super::SampleMode;
    use TFeature;

    #[test]
    fn test_sampler_nonblocking() {
        let exp_name = "test_sampler_nonblocking";
        let (gather_sender, gather_receiver) = channel::bounded(10, "gather-samples");
        let model = Arc::new(RwLock::new((Model::new(1), "test-model".to_string())));
        let gatherer = Gatherer::new(gather_receiver, 100, model.clone(), exp_name.to_string());
        gatherer.run(SampleMode::LOCAL);

        let mut examples: Vec<ExampleWithScore> = vec![];
        for i in 0..100 {
            let t = get_example(vec![i as TFeature, 1, 2], 0.0);
            gather_sender.send(((t.clone(), 1), 1));
            examples.push(t);
        }
        sleep(Duration::from_millis(1000));  // wait for the gatherer releasing the new sample
        let sample_model = load_sample_local(exp_name);
        let mut sample = sample_model.unwrap().1;
        sample.sort_by(|t1, t2| (t1.0).feature[0].partial_cmp(&(t2.0).feature[0]).unwrap());
        for (input, output) in examples.iter().zip(sample.iter()) {
            assert_eq!(*input, *output);
        }
    }

    fn get_example(features: Vec<TFeature>, score: f32) -> ExampleWithScore {
        let label: i8 = -1;
        let example = LabeledData::new(features, label);
        (example, (score, 0))
    }
}
*/