
use std::thread::spawn;

use commons::channel::Receiver;
use commons::performance_monitor::PerformanceMonitor;

use super::LockedBuffer;
use super::SampleMode;
use super::io::load_local;
use super::io::load_s3;


pub struct Loader {
    new_sample_buffer: LockedBuffer,
    signal_channel:    Receiver<String>,
}


impl Loader {
    /// * `new_sample_buffer`: the reference to the alternate memory buffer of the buffer loader
    /// * `signal_channel`: the channel to receive the parameters on from where to load examples
    pub fn new(
        new_sample_buffer: LockedBuffer,
        signal_channel:    Receiver<String>,
    ) -> Loader {
        Loader {
            new_sample_buffer: new_sample_buffer,
            signal_channel:    signal_channel,
        }
    }

    /// Start the gatherer.
    ///
    /// Fill the alternate memory buffer of the buffer loader
    pub fn run(&self, mode: SampleMode) {
        let signal_channel = self.signal_channel.clone();
        let buffer: LockedBuffer = self.new_sample_buffer.clone();
        info!("Starting non-blocking loader");
        spawn(move || {
            loop {
                match mode {
                    SampleMode::LOCAL => {
                        loader(
                            signal_channel.clone(),
                            buffer.clone(),
                            &load_local,
                        );
                    },
                    SampleMode::S3 => {
                        loader(
                            signal_channel.clone(),
                            buffer.clone(),
                            &load_s3,
                        );
                    },
                    // SampleMode.MEMORY should be handled by `gatherer`
                    _ => {
                        error!("Unrecognized Sample Mode in the Loader");
                    },
                }
            }
        });
    }
}


fn loader(
    signal_channel: Receiver<String>,
    new_sample_buffer: LockedBuffer,
    handler: &Fn(String, LockedBuffer) -> (),
) {
    let mut pm = PerformanceMonitor::new();
    pm.start();
    if let Some(signal) = signal_channel.recv() {
        handler(signal.to_string(), new_sample_buffer);
    }
    debug!("sample-loader, {}", pm.get_duration());
}
