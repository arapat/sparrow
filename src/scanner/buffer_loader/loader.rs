
use std::thread::spawn;
use std::thread::sleep;
use std::time::Duration;

use SampleMode;
use commons::persistent_io::load_local;
use commons::persistent_io::load_s3;
use commons::performance_monitor::PerformanceMonitor;
use commons::persistent_io::VersionedSampleModel;

use super::LockedBuffer;
use super::LockedModelBuffer;


pub struct Loader {
    new_sample_buffer: LockedBuffer,
    new_model_buffer:  LockedModelBuffer,
    sleep_duration:    u64,
    exp_name:          String,
}


impl Loader {
    /// * `new_sample_buffer`: the reference to the alternate memory buffer of the buffer loader
    pub fn new(
        new_sample_buffer: LockedBuffer,
        new_model_buffer: LockedModelBuffer,
        sleep_duration: usize,
        exp_name:       String,
    ) -> Loader {
        Loader {
            new_sample_buffer: new_sample_buffer,
            new_model_buffer:  new_model_buffer,
            sleep_duration:    sleep_duration as u64,
            exp_name:          exp_name,
        }
    }

    /// Start the loaders
    ///
    /// Fill the alternate memory buffer of the buffer loader
    pub fn run(&self, mode: SampleMode) {
        let buffer: LockedBuffer = self.new_sample_buffer.clone();
        let model_buffer: LockedModelBuffer = self.new_model_buffer.clone();
        let sleep_duration = self.sleep_duration;
        let exp_name = self.exp_name.clone();
        info!("Starting non-blocking loader");
        spawn(move || {
            let mut last_version: usize = 0;
            loop {
                last_version = match mode {
                    SampleMode::LOCAL => {
                        loader(
                            buffer.clone(),
                            model_buffer.clone(),
                            load_local,
                            last_version,
                            exp_name.as_str(),
                        )
                    },
                    SampleMode::S3 => {
                        loader(
                            buffer.clone(),
                            model_buffer.clone(),
                            load_s3,
                            last_version,
                            exp_name.as_str(),
                        )
                    },
                };
                sleep(Duration::from_secs(sleep_duration));
            }
        });
    }
}


fn loader<F>(
    new_sample_buffer: LockedBuffer,
    new_model_buffer: LockedModelBuffer,
    handler: F,
    last_version: usize,
    exp_name: &str,
) -> usize
where F: Fn(usize, &str) -> Option<VersionedSampleModel> {
    let mut pm = PerformanceMonitor::new();
    pm.start();
    let ret = handler(last_version, exp_name);
    if ret.is_none() {
        debug!("scanner, failed to receive a new sample");
        last_version
    } else {
        let (version, new_sample, new_model) = ret.unwrap();
        let new_sample_lock = new_sample_buffer.write();
        *(new_sample_lock.unwrap()) = Some((version, new_sample));
        let new_model_lock = new_model_buffer.write();
        *(new_model_lock.unwrap()) = Some(new_model);
        debug!("scanner, received a new sample, {}, {}", version, pm.get_duration());
        version
    }
}