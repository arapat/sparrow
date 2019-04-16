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


type VersionedSample = (usize, Vec<ExampleWithScore>);


const FILENAME: &str = "sample.bin";
const REGION:   &str = "us-west-1";
const BUCKET:   &str = "tmsn-cache";
const S3_PATH:  &str = "samples/";


// For gatherer

// also do the job of `load_memory`
pub fn write_memory(
    new_sample: Vec<ExampleWithScore>,
    new_sample_buffer: LockedBuffer,
    _version: usize,
) {
    let new_sample_lock = new_sample_buffer.write();
    *(new_sample_lock.unwrap()) = Some(new_sample);
}


pub fn write_local(
    new_sample: Vec<ExampleWithScore>,
    _new_sample_buffer: LockedBuffer,
    version: usize,
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
) {
    let region = REGION.parse().unwrap();
    // TODO: Add support to read credentials from the config file
    // Read credentials from the environment variables
    let credentials = Credentials::default();
    let bucket = Bucket::new(BUCKET, region, credentials).unwrap();
    let filename = S3_PATH.to_string() + FILENAME;

    // In case the file exists, delete it
    let (_, _) = bucket.delete(&filename).unwrap();
    let data: VersionedSample = (version, new_sample);
    let (_, code) = bucket.put(
        &filename, &serialize(&data).unwrap(), "application/octet-stream").unwrap();
    assert_eq!(200, code);
}


// For loader

pub fn load_local(
    new_sample_buffer: LockedBuffer,
    last_version: usize,
) -> usize {
    let ori_filename = FILENAME.to_string();
    let filename = ori_filename.clone() + "_READING";
    if rename(ori_filename, filename.clone()).is_ok() {
        let (version, new_sample): VersionedSample =
            deserialize(read_all(&filename).as_ref()).unwrap();
        if version > last_version {
            let new_sample_lock = new_sample_buffer.write();
            *(new_sample_lock.unwrap()) = Some(new_sample);
            remove_file(filename).unwrap();
            return version;
        }
    }
    last_version
}


pub fn load_s3(
    new_sample_buffer: LockedBuffer,
    last_version: usize,
) -> usize {
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
        let (version, data) = deserialize(&data).unwrap();
        if version > last_version {
            *(new_sample_lock.unwrap()) = Some(data);
            return version;
        }
    }
    last_version
}
