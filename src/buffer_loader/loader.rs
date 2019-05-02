
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
}


impl Loader {
    /// * `new_sample_buffer`: the reference to the alternate memory buffer of the buffer loader
    pub fn new(
        new_sample_buffer: LockedBuffer,
        sleep_duration: usize,
    ) -> Loader {
        Loader {
            new_sample_buffer: new_sample_buffer,
            sleep_duration:    sleep_duration as u64,
        }
    }

    /// Start the gatherer.
    ///
    /// Fill the alternate memory buffer of the buffer loader
    pub fn run(&self, mode: SampleMode) {
        let buffer: LockedBuffer = self.new_sample_buffer.clone();
        let sleep_duration = self.sleep_duration;
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
                        )
                    },
                    SampleMode::S3 => {
                        loader(
                            buffer.clone(),
                            &load_s3,
                            last_version,
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
    handler: &Fn(LockedBuffer, usize) -> Option<usize>,
    last_version: usize,
) -> usize {
    let mut pm = PerformanceMonitor::new();
    pm.start();
    let version = handler(new_sample_buffer, last_version);
    if version.is_none() {
        last_version
    } else {
        let v = version.unwrap();
        debug!("sample-loader, {}, {}", v, pm.get_duration());
        v
    }
}
