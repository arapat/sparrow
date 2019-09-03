
use rand::Rng;

use std::fs::rename;
use std::sync::Arc;
use std::sync::RwLock;
use std::thread::spawn;
use rand::thread_rng;

use bincode::serialize;
use commons::channel::Receiver;
use commons::io::write_all;
use commons::ExampleWithScore;
use commons::performance_monitor::PerformanceMonitor;

use super::LockedBuffer;
use super::SampleMode;
use super::io::write_memory;
use super::io::write_local;
use super::io::write_s3;


pub struct Gatherer {
    gather_new_sample:      Receiver<((ExampleWithScore, u32), u32)>,
    new_sample_capacity:    usize,
    new_sample_buffer:      LockedBuffer,
    current_sample_version: Arc<RwLock<usize>>,
    exp_name:               String,
}


impl Gatherer {
    /// * `new_sample_buffer`: the reference to the alternate memory buffer of the buffer loader
    /// * `new_sample_capacity`: the size of the memory buffer of the buffer loader
    pub fn new(
        gather_new_sample:      Receiver<((ExampleWithScore, u32), u32)>,
        new_sample_capacity:    usize,
        new_sample_buffer:      LockedBuffer,
        current_sample_version: Arc<RwLock<usize>>,
        exp_name:               String,
    ) -> Gatherer {
        Gatherer {
            gather_new_sample:   gather_new_sample,
            new_sample_capacity: new_sample_capacity,
            new_sample_buffer:   new_sample_buffer,
            current_sample_version: current_sample_version,
            exp_name:            exp_name,
        }
    }

    /// Start the gatherer.
    ///
    /// Fill the alternate memory buffer of the buffer loader
    pub fn run(&self, mode: SampleMode) {
        let new_sample_capacity = self.new_sample_capacity;
        let gather_new_sample = self.gather_new_sample.clone();
        let new_sample_buffer = self.new_sample_buffer.clone();
        let current_sample_version = self.current_sample_version.clone();
        let exp_name = self.exp_name.clone();
        info!("Starting non-blocking gatherer");
        spawn(move || {
            let mut version = 0;
            loop {
                version += 1;
                match mode {
                    SampleMode::MEMORY => {
                        gather(
                            new_sample_capacity,
                            new_sample_buffer.clone(),
                            gather_new_sample.clone(),
                            write_memory,
                            version,
                            exp_name.as_str(),
                        );
                    },
                    SampleMode::LOCAL => {
                        gather(
                            new_sample_capacity,
                            new_sample_buffer.clone(),
                            gather_new_sample.clone(),
                            write_local,
                            version,
                            exp_name.as_str(),
                        );
                    },
                    SampleMode::S3 => {
                        gather(
                            new_sample_capacity,
                            new_sample_buffer.clone(),
                            gather_new_sample.clone(),
                            write_s3,
                            version,
                            exp_name.as_str(),
                        );
                    },
                }
                {
                    *(current_sample_version.write().unwrap()) = version;
                }
            }
        });
    }
}


fn gather<F>(
    new_sample_capacity: usize,
    new_sample_buffer: LockedBuffer,
    gather_new_sample: Receiver<((ExampleWithScore, u32), u32)>,
    handler: F,
    version: usize,
    exp_name: &str,
) where F: Fn(Vec<ExampleWithScore>, LockedBuffer, usize, &str) {
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
        if new_sample.len() - last_log_count > new_sample_capacity / 10 {
            debug!("sampler, progress, {}", new_sample.len());
            last_log_count = new_sample.len();
        }
    }
    thread_rng().shuffle(&mut new_sample);
    debug!("sampler, finished, generate sample, {}, {}, {}, {}, {}",
           total_scanned, new_sample.len(), num_total_positive, num_unique, num_unique_positive);
    // Create a snapshot for continous training
    let filename = "latest_sample.bin".to_string() + "_WRITING";
    write_all(&filename, &serialize(&(version, new_sample.clone())).unwrap())
        .expect("Failed to write the sample set to file for snapshot");
    rename(filename, "latest_sample.bin".to_string()).unwrap();
    // Send the sample to the handler
    handler(new_sample, new_sample_buffer, version, exp_name);
    let duration = pm.get_duration();
    debug!("sample-gatherer, {}, {}", duration, new_sample_capacity as f32 / duration);
}


#[cfg(test)]
mod tests {
    use std::thread::sleep;

    use std::sync::Arc;
    use std::sync::RwLock;
    use std::time::Duration;

    use commons::channel;
    use commons::ExampleWithScore;
    use labeled_data::LabeledData;
    use super::Gatherer;
    use super::super::SampleMode;
    use ::TFeature;

    #[test]
    fn test_sampler_nonblocking() {
        let (gather_sender, gather_receiver) = channel::bounded(10, "gather-samples");
        let mem_buffer = Arc::new(RwLock::new(None));
        let gatherer = Gatherer::new(gather_receiver, 100, mem_buffer.clone());
        gatherer.run(SampleMode::MEMORY);

        let mut examples: Vec<ExampleWithScore> = vec![];
        for i in 0..100 {
            let t = get_example(vec![i as TFeature, 1, 2], 0.0);
            gather_sender.send(((t.clone(), 1), 1));
            examples.push(t);
        }
        sleep(Duration::from_millis(1000));  // wait for the gatherer releasing the new sample
        let mut all_sampled: Vec<_> = mem_buffer.write().unwrap().take().unwrap();
        all_sampled.sort_by(|t1, t2| (t1.0).feature[0].partial_cmp(&(t2.0).feature[0]).unwrap());
        for (input, output) in examples.iter().zip(all_sampled.iter()) {
            assert_eq!(*input, *output);
        }
    }

    fn get_example(features: Vec<TFeature>, score: f32) -> ExampleWithScore {
        let label: i8 = -1;
        let example = LabeledData::new(features, label);
        (example, (score, 0))
    }
}