use std::sync::mpsc::Receiver;
use std::thread::spawn;
use std::thread::sleep;
use std::time::Duration;

use SampleMode;
use commons::persistent_io::load_sample;
use commons::persistent_io::load_sample_local;
use commons::persistent_io::load_sample_s3;

use super::LockedBuffer;


pub struct Loader {
    new_buffer: LockedBuffer,
    sleep_duration:    u64,
    exp_name:          String,
}


impl Loader {
    /// * `new_sample_buffer`: the reference to the alternate memory buffer of the buffer loader
    pub fn new(
        new_buffer: LockedBuffer,
        sleep_duration: usize,
        exp_name:       String,
    ) -> Loader {
        Loader {
            new_buffer:        new_buffer,
            sleep_duration:    sleep_duration as u64,
            exp_name:          exp_name,
        }
    }

    /// Start the loaders
    ///
    /// Fill the alternate memory buffer of the buffer loader
    pub fn run(&self, mode: SampleMode, sampler_signal_receiver: Receiver<usize>) {
        let buffer: LockedBuffer = self.new_buffer.clone();
        let sleep_duration = self.sleep_duration;
        let exp_name = self.exp_name.clone();
        let load_func = match mode {
            SampleMode::LOCAL => load_sample_local,
            SampleMode::S3 => load_sample_s3,
        };
        info!("Starting non-blocking loader");
        spawn(move || {
            for incoming_version in sampler_signal_receiver.iter() {
                debug!("buffer-loader, start download, {}", incoming_version);
                loop {
                    let sample = load_sample(load_func, exp_name.as_str());
                    if sample.is_some() {
                        let (version, new_sample, new_model) = sample.unwrap();
                        if version >= incoming_version {
                            let new_buffer = buffer.write();
                            *(new_buffer.unwrap()) = Some((version, new_sample, new_model));
                            debug!("buffer-loader, sample downloaded, {}", version);
                            break;
                        }
                    }
                    sleep(Duration::from_secs(sleep_duration));
                }
            }
        });
    }
}
