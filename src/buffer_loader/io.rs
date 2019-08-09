use std::fs::rename;
use std::fs::remove_file;

use bincode::deserialize;
use bincode::serialize;
use commons::ExampleWithScore;
use commons::io::read_all;
use commons::io::write_all;
use commons::io::load_s3 as io_load_s3;
use commons::io::write_s3 as io_write_s3;
use super::LockedBuffer;


type VersionedSample = (usize, Vec<ExampleWithScore>);


pub const FILENAME: &str = "sample.bin";
pub const REGION:   &str = "us-east-1";
pub const BUCKET:   &str = "tmsn-cache2";
pub const S3_PATH:  &str = "sparrow-samples/";


// For gatherer

// also do the job of `load_memory`
pub fn write_memory(
    new_sample: Vec<ExampleWithScore>,
    new_sample_buffer: LockedBuffer,
    version: usize,
    _exp_name: &str,
) {
    let new_sample_lock = new_sample_buffer.write();
    *(new_sample_lock.unwrap()) = Some((version, new_sample));
}


pub fn write_local(
    new_sample: Vec<ExampleWithScore>,
    _new_sample_buffer: LockedBuffer,
    version: usize,
    _exp_name: &str,
) {
    let filename = FILENAME.to_string() + "_WRITING";
    let data: VersionedSample = (version, new_sample);
    write_all(&filename, &serialize(&data).unwrap())
        .expect("Failed to write the sample set to file");
    rename(filename, FILENAME.to_string()).unwrap();
}


pub fn write_s3(
    new_sample: Vec<ExampleWithScore>,
    _new_sample_buffer: LockedBuffer,
    version: usize,
    exp_name: &str,
) {
    let data: VersionedSample = (version, new_sample);
    debug!("sampler, start, write new sample to s3");
    let s3_path = format!("{}/{}", exp_name, S3_PATH);
    io_write_s3(REGION, BUCKET, s3_path.as_str(), FILENAME, &serialize(&data).unwrap());
    debug!("sampler, finished, write new sample to s3");
    let filename = FILENAME.to_string() + "_WRITING";
    write_all(&filename, &serialize(&data).unwrap())
        .expect("Failed to write the sample set to file");
    rename(filename, format!("{}_{}", FILENAME, version)).unwrap();
}


// For loader

pub fn load_local(
    new_sample_buffer: LockedBuffer,
    last_version: usize,
    _exp_name: &str,
) -> Option<usize> {
    let ori_filename = FILENAME.to_string();
    let filename = ori_filename.clone() + "_READING";
    if rename(ori_filename, filename.clone()).is_ok() {
        let (version, new_sample): VersionedSample =
            deserialize(read_all(&filename).as_ref()).unwrap();
        if version > last_version {
            let new_sample_lock = new_sample_buffer.write();
            *(new_sample_lock.unwrap()) = Some((version, new_sample));
            remove_file(filename).unwrap();
            return Some(version);
        }
    }
    None
}


pub fn load_s3(
    new_sample_buffer: LockedBuffer,
    last_version: usize,
    exp_name: &str,
) -> Option<usize> {
    // debug!("scanner, start, download sample from s3");
    let s3_path = format!("{}/{}", exp_name, S3_PATH);
    let ret = io_load_s3(REGION, BUCKET, s3_path.as_str(), FILENAME);
    if ret.is_none() {
        return None;
    }
    let (data, code) = ret.unwrap();
    if code == 200 {
        let (version, data) = deserialize(&data).unwrap();
        if version > last_version {
            let new_sample_lock = new_sample_buffer.write();
            *(new_sample_lock.unwrap()) = Some((version, data));
            // debug!("scanner, finished, download sample from s3, succeed");
            return Some(version);
        }
        debug!("scanner, finished, download sample from s3, remote model is old, {}, {}",
               version, last_version);
    } else {
        debug!("scanner, failed, download sample from s3, err {}", code);
    }
    None
}
