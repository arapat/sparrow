use std::fs::rename;
use std::fs::remove_file;

use bincode::deserialize;
use bincode::serialize;
use commons::ExampleWithScore;
use commons::io::read_all;
use commons::io::write_all;
use commons::io::load_s3 as io_load_s3;
use commons::io::write_s3 as io_write_s3;
use commons::Model;


pub type VersionedSampleModel = (usize, Vec<ExampleWithScore>, Model);


pub const FILENAME: &str = "sample.bin";
pub const REGION:   &str = "us-east-1";
pub const BUCKET:   &str = "tmsn-cache2";
pub const S3_PATH:  &str = "sparrow-samples/";


// For gatherer

pub fn write_local(
    new_sample: Vec<ExampleWithScore>,
    model: Model,
    version: usize,
    _exp_name: &str,
) {
    let filename = FILENAME.to_string() + "_WRITING";
    let data: VersionedSampleModel = (version, new_sample, model);
    write_all(&filename, &serialize(&data).unwrap())
        .expect("Failed to write the sample set to file");
    rename(filename, FILENAME.to_string()).unwrap();
}


pub fn write_s3(
    new_sample: Vec<ExampleWithScore>,
    model: Model,
    version: usize,
    exp_name: &str,
) {
    let data: VersionedSampleModel = (version, new_sample, model);
    debug!("sampler, start, write new sample to s3, {}", version);
    let s3_path = format!("{}/{}", exp_name, S3_PATH);
    io_write_s3(REGION, BUCKET, s3_path.as_str(), FILENAME, &serialize(&data).unwrap());
    debug!("sampler, finished, write new sample to s3, {}", version);
    let filename = FILENAME.to_string() + "_WRITING";
    write_all(&filename, &serialize(&data).unwrap())
        .expect(format!("Failed to write the sample set to file, {}", version).as_str());
    rename(filename, format!("{}_{}", FILENAME, version)).unwrap();
}


// For loader

pub fn load_local(last_version: usize, _exp_name: &str) -> Option<VersionedSampleModel> {
    let ori_filename = FILENAME.to_string();
    let filename = ori_filename.clone() + "_READING";
    if rename(ori_filename, filename.clone()).is_ok() {
        let (version, sample, model): VersionedSampleModel =
            deserialize(read_all(&filename).as_ref()).unwrap();
        if version > last_version {
            remove_file(filename).unwrap();
            return Some((version, sample, model));
        }
    }
    None
}


pub fn load_s3(last_version: usize, exp_name: &str) -> Option<VersionedSampleModel> {
    // debug!("scanner, start, download sample from s3");
    let s3_path = format!("{}/{}", exp_name, S3_PATH);
    let ret = io_load_s3(REGION, BUCKET, s3_path.as_str(), FILENAME);
    if ret.is_none() {
        return None;
    }
    let (data, code) = ret.unwrap();
    if code == 200 {
        let (version, sample, model) = deserialize(&data).unwrap();
        if version > last_version {
            return Some((version, sample, model));
        }
        debug!("scanner, finished, download sample from s3, remote sample is old, {}, {}",
               version, last_version);
    } else {
        debug!("scanner, failed, download sample from s3, err {}", code);
    }
    None
}
