

/*
use std::str;

use s3::bucket::Bucket;
use s3::credentials::Credentials;
use s3::error::S3Result;

const REGION: &str = "us-east-1";
const BUCKET: &str = "drazen-test-bucket-2";

pub fn main() -> S3Result<()> {
    let region = REGION.parse()?;
//     Create Bucket in REGION for BUCKET
    let credentials = Credentials::default();
    let bucket = Bucket::new(BUCKET, region, credentials)?;

    // Make sure that our "test_file" doesn't exist, delete it if it does.
    bucket.delete("test_file")?;

    // Put a "test_file" with the contents of MESSAGE at the root of the
    // bucket.
    let (_, code) = bucket.put("test_file", MESSAGE.as_bytes(), "text/plain")?;
    assert_eq!(200, code);

    // Get the "test_file" contents and make sure that the returned message
    // matches what we sent.
    let (data, code) = bucket.get("test_file")?;
    let string = str::from_utf8(&data).unwrap();
    assert_eq!(200, code);
    assert_eq!(MESSAGE, string);
}
*/

use bincode::serialize;
use commons::ExampleWithScore;
use commons::io::write_all;
use super::LockedBuffer;


pub static FILENAME: &str = "sample.bin";


// For gatherer

pub fn write_memory(
    new_sample: Vec<ExampleWithScore>,
    new_sample_buffer: LockedBuffer,
) {
    let new_sample_lock = new_sample_buffer.write();
    *(new_sample_lock.unwrap()) = new_sample;
}


pub fn write_local(
    new_sample: Vec<ExampleWithScore>,
    new_sample_buffer: LockedBuffer,
) {
    let filename = FILENAME.to_string();
    write_all(&filename, &serialize(&new_sample).unwrap());
    // signal_channel.send(filename);
}


pub fn write_s3(
    new_sample: Vec<ExampleWithScore>,
    new_sample_buffer: LockedBuffer,
) {
    // TODO: upload S3 and send out S3 info
}


// For loader

pub fn load_local(
    filename: String,
    new_sample_lock: LockedBuffer,
) {
    // let new_sample_lock = new_sample_buffer.write();
    // *(new_sample_lock.unwrap()) = Some(new_sample);
}



pub fn load_s3(
    s3_info: String,
    new_sample_lock: LockedBuffer,
) {
    // TODO: Load from S3
}