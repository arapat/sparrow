use std::fs::rename;
use std::fs::remove_file;
use std::io::Write;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::mpsc::Receiver;

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
use commons::performance_monitor::PerformanceMonitor;
use commons::Model;


// (sample_version, new_sample, model, model_sig);
pub type VersionedSampleModel = (usize, Vec<ExampleWithScore>, Model, String);
pub type ModelPack = (Model, String, f32);
// LockedBuffer is set to None once it is read by the receiver
pub type LockedBuffer = Arc<RwLock<Option<VersionedSampleModel>>>;


const S3_PATH_SAMPLE:  &str = "sparrow-samples/";
const SAMPLE_FILENAME: &str = "sample.bin";
const S3_PATH_ASSIGNS: &str = "sparrow-assigns/";
const ASSIGN_FILENAME: &str = "assign.bin";
const S3_PATH_BINS:    &str = "sparrow-bins/";
const BINS_FILENAME:   &str = "bins.json";


// For gatherer

fn get_sample_local_filename(exp_name: &str) -> String {
    exp_name.to_string() + SAMPLE_FILENAME
}


pub fn write_sample_local(
    new_sample: Vec<ExampleWithScore>,
    model: Model,
    model_sig: String,
    version: usize,
    exp_name: &str,
) {
    let base_filename = get_sample_local_filename(exp_name);
    let temp_filename = base_filename.clone() + "_WRITING";
    let data: VersionedSampleModel = (version, new_sample, model, model_sig);
    write_all(&temp_filename, &serialize(&data).unwrap())
        .expect("Failed to write the sample set to file");
    rename(temp_filename, base_filename.to_string()).unwrap();
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

pub fn load_sample<F>(load_handler: F, exp_name: &str) -> Option<VersionedSampleModel>
where F: Fn(&str) -> Option<VersionedSampleModel> {
    let mut pm = PerformanceMonitor::new();
    pm.start();
    let ret = load_handler(exp_name);
    if ret.is_none() {
        debug!("scanner, failed to receive a sample");
        None
    } else {
        debug!("scanner, received a new sample");
        ret
    }
}


pub fn load_sample_local(exp_name: &str) -> Option<VersionedSampleModel> {
    let base_filename = get_sample_local_filename(exp_name);
    let temp_filename = base_filename.clone() + "_READING";
    if rename(base_filename, temp_filename.clone()).is_ok() {
        let (version, sample, model, model_sig): VersionedSampleModel =
            deserialize(read_all(&temp_filename).as_ref()).unwrap();
        remove_file(temp_filename).unwrap();
        return Some((version, sample, model, model_sig));
    }
    None
}


#[cfg(not(test))]
pub fn load_sample_s3(exp_name: &str) -> Option<VersionedSampleModel> {
    // debug!("scanner, start, download sample from s3");
    let s3_path = format!("{}/{}", exp_name, S3_PATH_SAMPLE);
    let ret = io_load_s3(REGION, BUCKET, s3_path.as_str(), SAMPLE_FILENAME);
    if ret.is_none() {
        return None;
    }
    let (data, code) = ret.unwrap();
    if code == 200 {
        let (version, sample, model, model_sig) = deserialize(&data).unwrap();
        return Some((version, sample, model, model_sig));
    } else {
        debug!("scanner, failed, download sample from s3, err {}", code);
    }
    None
}


#[cfg(test)]
pub fn load_sample_s3(_last_version: usize, _exp_name: &str) -> Option<VersionedSampleModel> {
    use commons::test_helper::get_n_random_examples;
    let model = Model::new(1);
    let examples = get_n_random_examples(1000, 20);
    Some((0, examples, model, "mock sample".to_string()))
}


// read/write model

#[cfg(not(test))]
pub fn write_model(model: &Model, timestamp: f32, save_process: bool) -> String {
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
    json
}

#[cfg(test)]
pub fn write_model(model: &Model, timestamp: f32, _save_process: bool) -> String {
    serde_json::to_string(&(timestamp, model.size(), model)).expect(
        "Local model cannot be serialized."
    )
}


// pub fn upload_model(
//     model: &Model, sig: &String, gamma: f32, exp_name: &String,
// ) -> bool {
//     let data: ModelPack = (model.clone(), sig.clone(), gamma);
//     let s3_path = format!("{}/{}", exp_name, S3_PATH_MODELS);
//     io_write_s3(REGION, BUCKET, s3_path.as_str(), MODEL_FILENAME, &serialize(&data).unwrap())
// }


pub fn read_model() -> (f32, usize, Model) {
    serde_json::from_str(&raw_read_all(&"model.json".to_string()))
            .expect(&format!("Cannot parse the model in `model.json`"))
}


// pub fn download_model(exp_name: &String) -> Option<ModelPack> {
//     // debug!("sampler, start, download model");
//     let s3_path = format!("{}/{}", exp_name, S3_PATH_MODELS);
//     let ret = io_load_s3(REGION, BUCKET, s3_path.as_str(), MODEL_FILENAME);
//     // debug!("sampler, finished, download model");
//     if ret.is_none() {
//         debug!("sample, download model, failed");
//         return None;
//     }
//     let (data, code) = ret.unwrap();
//     if code == 200 {
//         // debug!("sample, download model, succeed");
//         Some(deserialize(&data).unwrap())
//     } else {
//         debug!("sample, download model, failed with return code {}", code);
//         None
//     }
// }


// Read/write assignments

#[cfg(not(test))]
pub fn upload_assignments(worker_assign: &Vec<Option<usize>>, exp_name: &String) -> bool {
    let data = worker_assign;
    let s3_path = format!("{}/{}", exp_name, S3_PATH_ASSIGNS);
    io_write_s3(REGION, BUCKET, s3_path.as_str(), ASSIGN_FILENAME, &serialize(&data).unwrap())
}


#[cfg(test)]
pub fn upload_assignments(_worker_assign: &Vec<Option<usize>>, _exp_name: &String) -> bool {
    true
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