use std::fs::rename;
use std::fs::remove_file;
use std::io::Write;

use REGION;
use BUCKET;
use bincode::deserialize;
use bincode::serialize;
use commons::ExampleWithScore;
use commons::bins::Bins;
use commons::io::read_all;
use commons::io::write_all;
use commons::io::create_bufwriter;
use commons::io::raw_read_all;
use commons::io::load_s3 as io_load_s3;
use commons::io::write_s3 as io_write_s3;
use commons::Model;


pub type VersionedSampleModel = (usize, Vec<ExampleWithScore>, Model, String);


const S3_PATH_SAMPLE:  &str = "sparrow-samples/";
const SAMPLE_FILENAME: &str = "sample.bin";
const S3_PATH_MODELS:  &str = "sparrow-models/";
const MODEL_FILENAME:  &str = "model.bin";
const S3_PATH_ASSIGNS: &str = "sparrow-assigns/";
const ASSIGN_FILENAME: &str = "assign.bin";
const S3_PATH_BINS:    &str = "sparrow-bins/";
const BINS_FILENAME:   &str = "bins.json";


// For gatherer

pub fn write_sample_local(
    new_sample: Vec<ExampleWithScore>,
    model: Model,
    model_sig: String,
    version: usize,
    _exp_name: &str,
) {
    let filename = SAMPLE_FILENAME.to_string() + "_WRITING";
    let data: VersionedSampleModel = (version, new_sample, model, model_sig);
    write_all(&filename, &serialize(&data).unwrap())
        .expect("Failed to write the sample set to file");
    rename(filename, SAMPLE_FILENAME.to_string()).unwrap();
}


pub fn write_sample_s3(
    new_sample: Vec<ExampleWithScore>,
    model: Model,
    model_sig: String,
    version: usize,
    exp_name: &str,
) {
    let data: VersionedSampleModel = (version, new_sample, model, model_sig);
    debug!("sampler, start, write new sample to s3, {}", version);
    let s3_path = format!("{}/{}", exp_name, S3_PATH_SAMPLE);
    io_write_s3(REGION, BUCKET, s3_path.as_str(), SAMPLE_FILENAME, &serialize(&data).unwrap());
    debug!("sampler, finished, write new sample to s3, {}", version);
    let filename = SAMPLE_FILENAME.to_string() + "_WRITING";
    write_all(&filename, &serialize(&data).unwrap())
        .expect(format!("Failed to write the sample set to file, {}", version).as_str());
    rename(filename, format!("{}_{}", SAMPLE_FILENAME, version)).unwrap();
}


// For loader

pub fn load_sample_local(last_version: usize, _exp_name: &str) -> Option<VersionedSampleModel> {
    let ori_filename = SAMPLE_FILENAME.to_string();
    let filename = ori_filename.clone() + "_READING";
    if rename(ori_filename, filename.clone()).is_ok() {
        let (version, sample, model, model_sig): VersionedSampleModel =
            deserialize(read_all(&filename).as_ref()).unwrap();
        if version > last_version {
            remove_file(filename).unwrap();
            return Some((version, sample, model, model_sig));
        }
    }
    None
}


pub fn load_sample_s3(last_version: usize, exp_name: &str) -> Option<VersionedSampleModel> {
    // debug!("scanner, start, download sample from s3");
    let s3_path = format!("{}/{}", exp_name, S3_PATH_SAMPLE);
    let ret = io_load_s3(REGION, BUCKET, s3_path.as_str(), SAMPLE_FILENAME);
    if ret.is_none() {
        return None;
    }
    let (data, code) = ret.unwrap();
    if code == 200 {
        let (version, sample, model, model_sig) = deserialize(&data).unwrap();
        if version > last_version {
            return Some((version, sample, model, model_sig));
        }
        debug!("scanner, finished, download sample from s3, remote sample is old, {}, {}",
               version, last_version);
    } else {
        debug!("scanner, failed, download sample from s3, err {}", code);
    }
    None
}

// read/write model

pub fn write_model(model: &Model, timestamp: f32, save_process: bool) {
    let json = serde_json::to_string(&(timestamp, model.size(), model)).expect(
        "Local model cannot be serialized."
    );
    let filename = {
        if save_process {
            format!("models/model_{}-v{}.json", model.size(), model.size())
        } else {
            "model.json".to_string()
        }
    };
    create_bufwriter(&filename).write(json.as_ref()).unwrap();
}


pub fn upload_model(
    model: &Model, sig: &String, gamma: f32, root_gamma: f32, exp_name: &String,
) -> bool {
    let data: (Model, String, f32, f32) = (model.clone(), sig.clone(), gamma, root_gamma);
    let s3_path = format!("{}/{}", exp_name, S3_PATH_MODELS);
    io_write_s3(REGION, BUCKET, s3_path.as_str(), MODEL_FILENAME, &serialize(&data).unwrap())
}


pub fn read_model() -> (f32, usize, Model) {
    serde_json::from_str(&raw_read_all(&"model.json".to_string()))
            .expect(&format!("Cannot parse the model in `model.json`"))
}


pub fn download_model(exp_name: &String) -> Option<(Model, String, f32, f32)> {
    // debug!("sampler, start, download model");
    let s3_path = format!("{}/{}", exp_name, S3_PATH_MODELS);
    let ret = io_load_s3(REGION, BUCKET, s3_path.as_str(), MODEL_FILENAME);
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


// Read/write assignments

pub fn upload_assignments(worker_assign: &Vec<Option<usize>>, exp_name: &String) -> bool {
    let data = worker_assign;
    let s3_path = format!("{}/{}", exp_name, S3_PATH_ASSIGNS);
    io_write_s3(REGION, BUCKET, s3_path.as_str(), ASSIGN_FILENAME, &serialize(&data).unwrap())
}


pub fn download_assignments(exp_name: &String) -> Option<Vec<Option<usize>>> {
    let s3_path = format!("{}/{}", exp_name, S3_PATH_ASSIGNS);
    let ret = io_load_s3(REGION, BUCKET, s3_path.as_str(), ASSIGN_FILENAME);
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


// Read/write bins

pub fn write_bins_disk(bins: &Vec<Bins>) {
    let mut file_buffer = create_bufwriter(&"models/bins.json".to_string());
    let json = serde_json::to_string(bins).expect("Bins cannot be serialized.");
    file_buffer.write(json.as_ref()).unwrap();
}


pub fn write_bins_s3(bins: &Vec<Bins>, exp_name: &String) {
    let s3_path = format!("{}/{}", exp_name, S3_PATH_BINS);
    io_write_s3(REGION, BUCKET, s3_path.as_str(), BINS_FILENAME, &serialize(&bins).unwrap());
}


pub fn read_bins_disk() -> Vec<Bins> {
    serde_json::from_str(&raw_read_all(&"models/bins.json".to_string()))
        .expect(&format!("Cannot parse the bins.json"))
}


pub fn read_bins_s3(exp_name: &String) -> Vec<Bins> {
    let s3_path = format!("{}/{}", exp_name, S3_PATH_BINS);
    let mut ret = io_load_s3(REGION, BUCKET, s3_path.as_str(), BINS_FILENAME);
    while ret.is_none() {
        ret = io_load_s3(REGION, BUCKET, s3_path.as_str(), BINS_FILENAME);
    }
    let (bins, _) = ret.unwrap();
    deserialize(&bins).unwrap()
}