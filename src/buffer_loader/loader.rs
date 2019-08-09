
use std::thread::spawn;
use std::thread::sleep;
use std::time::Duration;

use commons::performance_monitor::PerformanceMonitor;

use super::LockedBuffer;
use super::SampleMode;
use super::io::load_local;
use super::io::load_s3;


pub struct Loader {
    new_sample_buffer: LockedBuffer,
    sleep_duration:    u64,
    exp_name:          String,
}


impl Loader {
    /// * `new_sample_buffer`: the reference to the alternate memory buffer of the buffer loader
    pub fn new(
        new_sample_buffer: LockedBuffer,
        sleep_duration: usize,
        exp_name:       String,
    ) -> Loader {
        Loader {
            new_sample_buffer: new_sample_buffer,
            sleep_duration:    sleep_duration as u64,
            exp_name:          exp_name,
        }
    }

    /// Start the gatherer.
    ///
    /// Fill the alternate memory buffer of the buffer loader
    pub fn run(&self, mode: SampleMode) {
        let buffer: LockedBuffer = self.new_sample_buffer.clone();
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
                            &load_local,
                            last_version,
                            exp_name.as_str(),
                        )
                    },
                    SampleMode::S3 => {
                        loader(
                            buffer.clone(),
                            &load_s3,
                            last_version,
                            exp_name.as_str(),
                        )
                    },
                    // SampleMode.MEMORY should be handled by `gatherer`
                    SampleMode::MEMORY => {
                        last_version
                    },
                };
                sleep(Duration::from_secs(sleep_duration));
            }
        });
    }
}


fn loader(
    new_sample_buffer: LockedBuffer,
    handler: &Fn(LockedBuffer, usize, &str) -> Option<usize>,
    last_version: usize,
    exp_name: &str,
) -> usize {
    let mut pm = PerformanceMonitor::new();
    pm.start();
    let version = handler(new_sample_buffer, last_version, exp_name);
    if version.is_none() {
        debug!("scanner, failed to receive a new sample");
        last_version
    } else {
        let v = version.unwrap();
        debug!("scanner, received a new sample, {}, {}", v, pm.get_duration());
        v
    }
}
