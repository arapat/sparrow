
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
}


impl Loader {
    /// * `new_sample_buffer`: the reference to the alternate memory buffer of the buffer loader
    pub fn new(
        new_sample_buffer: LockedBuffer,
    ) -> Loader {
        Loader {
            new_sample_buffer: new_sample_buffer,
        }
    }

    /// Start the gatherer.
    ///
    /// Fill the alternate memory buffer of the buffer loader
    pub fn run(&self, mode: SampleMode) {
        let buffer: LockedBuffer = self.new_sample_buffer.clone();
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
                sleep(Duration::from_secs(300));
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
