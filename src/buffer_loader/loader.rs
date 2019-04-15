
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
            loop {
                match mode {
                    SampleMode::LOCAL => {
                        loader(
                            buffer.clone(),
                            &load_local,
                        );
                    },
                    SampleMode::S3 => {
                        loader(
                            buffer.clone(),
                            &load_s3,
                        );
                    },
                    // SampleMode.MEMORY should be handled by `gatherer`
                    SampleMode::MEMORY => {
                    },
                }
                sleep(Duration::from_secs(sleep_duration));
            }
        });
    }
}


fn loader(
    new_sample_buffer: LockedBuffer,
    handler: &Fn(LockedBuffer) -> (),
) {
    let mut pm = PerformanceMonitor::new();
    pm.start();
    handler(new_sample_buffer);
    debug!("sample-loader, {}", pm.get_duration());
}
