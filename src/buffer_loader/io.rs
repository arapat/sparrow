use std::fs::rename;
use std::fs::remove_file;
use s3::bucket::Bucket;
use s3::credentials::Credentials;

use bincode::deserialize;
use bincode::serialize;
use commons::ExampleWithScore;
use commons::io::read_all;
use commons::io::write_all;
use super::LockedBuffer;


const FILENAME: &str = "sample.bin";
const REGION:   &str = "us-west-1";
const BUCKET:   &str = "tmsn-cache";
const S3_PATH:  &str = "samples/";


// For gatherer

pub fn write_memory(
    new_sample: Vec<ExampleWithScore>,
    new_sample_buffer: LockedBuffer,
) {
    let new_sample_lock = new_sample_buffer.write();
    *(new_sample_lock.unwrap()) = Some(new_sample);
}


pub fn write_local(
    new_sample: Vec<ExampleWithScore>,
    _new_sample_buffer: LockedBuffer,
) {
    let filename = FILENAME.to_string() + "_WRITING";
    write_all(&filename, &serialize(&new_sample).unwrap())
        .expect("Failed to write the sample set to file");
    rename(filename, FILENAME.to_string()).unwrap();
}


pub fn write_s3(
    new_sample: Vec<ExampleWithScore>,
    _new_sample_buffer: LockedBuffer,
) {
    let region = REGION.parse().unwrap();
    // TODO: Add support to read credentials from the config file
    // Read credentials from the environment variables
    let credentials = Credentials::default();
    let bucket = Bucket::new(BUCKET, region, credentials).unwrap();
    let filename = S3_PATH.to_string() + FILENAME;

    // In case the file exists, delete it
    let (_, _) = bucket.delete(&filename).unwrap();
    let (_, code) = bucket.put(
        &filename, &serialize(&new_sample).unwrap(), "application/octet-stream").unwrap();
    assert_eq!(200, code);
}


// For loader

pub fn load_local(
    new_sample_buffer: LockedBuffer,
) {
    let ori_filename = FILENAME.to_string();
    let filename = ori_filename.clone() + "_READING";
    if rename(ori_filename, filename.clone()).is_ok() {
        let new_sample: Vec<ExampleWithScore> = deserialize(read_all(&filename).as_ref()).unwrap();
        let new_sample_lock = new_sample_buffer.write();
        *(new_sample_lock.unwrap()) = Some(new_sample);
        remove_file(filename).unwrap();
    }
}



pub fn load_s3(
    new_sample_buffer: LockedBuffer,
) {
    let region = REGION.parse().unwrap();
    // TODO: Add support to read credentials from the config file
    // Read credentials from the environment variables
    let credentials = Credentials::default();
    let bucket = Bucket::new(BUCKET, region, credentials).unwrap();
    let mut filename = S3_PATH.to_string();
    filename.push_str(FILENAME);

    let (data, code) = bucket.get(&filename).unwrap();
    if code == 200 {
        let new_sample_lock = new_sample_buffer.write();
        *(new_sample_lock.unwrap()) = Some(deserialize(&data).unwrap());
    }
}
