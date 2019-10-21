use REGION;
use BUCKET;
use commons::Model;
use commons::io::load_s3;
use commons::io::write_s3;

use bincode::serialize;
use bincode::deserialize;

pub const S3_PATH_MODELS:  &str = "sparrow-models/";
pub const MODEL_FILENAME: &str = "model.bin";
pub const S3_PATH_ASSIGNS:  &str = "sparrow-assigns/";
pub const ASSIGN_FILENAME: &str = "assign.bin";


// Worker download models
pub fn download_model(exp_name: &String) -> Option<(Model, String, f32, f32)> {
    // debug!("sampler, start, download model");
    let s3_path = format!("{}/{}", exp_name, S3_PATH_MODELS);
    let ret = load_s3(REGION, BUCKET, s3_path.as_str(), MODEL_FILENAME);
    // debug!("sampler, finished, download model");
    if ret.is_none() {
        debug!("sample, download model, failed");
        return None;
    }
    let (data, code) = ret.unwrap();
    if code == 200 {
        // debug!("sample, download model, succeed");
        Some(deserialize(&data).unwrap())
    } else {
        debug!("sample, download model, failed with return code {}", code);
        None
    }
}


pub fn download_assignments(exp_name: &String) -> Option<Vec<Option<usize>>> {
    let s3_path = format!("{}/{}", exp_name, S3_PATH_ASSIGNS);
    let ret = load_s3(REGION, BUCKET, s3_path.as_str(), ASSIGN_FILENAME);
    // debug!("model sync, finished, download assignments");
    if ret.is_none() {
        // debug!("model sync, download assignments, failed");
        return None;
    }
    let (data, code) = ret.unwrap();
    if code == 200 {
        // debug!("model sync, download assignments, succeed");
        Some(deserialize(&data).unwrap())
    } else {
        debug!("model sync, download assignments, failed with return code {}", code);
        None
    }
}


// Server upload models
pub fn upload_model(
    model: &Model, sig: &String, gamma: f32, root_gamma: f32, exp_name: &String,
) -> bool {
    let data: (Model, String, f32, f32) = (model.clone(), sig.clone(), gamma, root_gamma);
    let s3_path = format!("{}/{}", exp_name, S3_PATH_MODELS);
    write_s3(REGION, BUCKET, s3_path.as_str(), MODEL_FILENAME, &serialize(&data).unwrap())
}


// Server upload assignments
pub fn upload_assignments(worker_assign: &Vec<Option<usize>>, exp_name: &String) -> bool {
    let data = worker_assign;
    let s3_path = format!("{}/{}", exp_name, S3_PATH_ASSIGNS);
    write_s3(REGION, BUCKET, s3_path.as_str(), ASSIGN_FILENAME, &serialize(&data).unwrap())
}