#[macro_use] extern crate serde_derive;
#[macro_use] extern crate log;
extern crate env_logger;
extern crate bincode;
extern crate bufstream;
extern crate ordered_float;
extern crate rand;
extern crate rayon;
extern crate serde_json;
extern crate time;


mod bins;
mod commons;
mod config;
mod data_loader; 
mod labeled_data;
mod learner;
mod tree;
mod validator;
mod boosting;
mod network;

use std::env;
use std::io::Read;
use std::io::Write;
use time::get_time;

use validator::get_adaboost_loss;
use validator::get_auprc;
use validator::validate;
use data_loader::io::create_bufreader;
use data_loader::io::read_k_lines;

use boosting::Boosting;
use commons::LossFunc;
use commons::Model;
use config::Config;
use data_loader::Format;
use data_loader::DataLoader;

use commons::get_symmetric_label;

fn main() {
    env_logger::Builder::from_default_env()
        .format(|buf, record| {
            let timestamp = get_time();
            let epoch_since_apr18: i64 = timestamp.sec - 1522540800;
            let formatted_ts = format!("{}.{:09}", epoch_since_apr18, timestamp.nsec);
            writeln!(
                buf, "{}, {}, {}, {}",
                record.level(), formatted_ts, record.module_path().unwrap(), record.args()
            )
        })
        .init();

    // Read configurations
    let mut reader = create_bufreader(&String::from("config.json"));
    let mut json = String::new();
    reader.read_to_string(&mut json).unwrap();
    let config: Config = serde_json::from_str(&json).expect(
        &format!("Cannot parse the config file.")
    );

    let mut home_dir = config.get_data_dir().clone();
    let remote_ips = config.get_network();
    if !home_dir.ends_with("/") {
        home_dir.push('/');
    }

    // read from text
    // let home_dir = std::env::home_dir().unwrap().display().to_string() +
    //                "/Downloads/splice/";
    // let training_data = home_dir.clone() + "training-shuffled.txt";
    // let testing_data =  home_dir.clone() + "testing-shuffled.txt";

    // read from bin
    let training_data = home_dir.clone() + "training.bin";
    let testing_data = home_dir.clone() + "testing.bin";

    let training_size = 50000000;
    let testing_size = 4627840;

    // use testing for training
    // let training_size = 4627840;
    // --> Text
    // let training_data = home_dir + "testing-shuffled.txt";
    // --> Binary
    // let training_data = home_dir.clone() + "testing.bin";

    let feature_size = 564;
    let batch_size = 10000;

    let max_sample_size = 100000;
    let max_bin_size = 2;
    let sample_ratio = 0.05;
    let ess_threshold = 0.5;
    let default_rho_gamma = 0.25;
    let eval_funcs: Vec<&LossFunc> = vec![
        &get_adaboost_loss,
        &get_auprc
    ];
    let max_trials_before_shrink = if training_size < 1000000 {
        training_size as u32
    } else {
        1000000
    };
    let validate_interval = 0;

    // debugging
    // let training_size = 100000;
    // let testing_size = 100000;
    // let testing_data = home_dir.clone() + "training.bin";
    // let sample_ratio = 0.8;
    // let validate_interval = 1;

    let training_loader = DataLoader::from_scratch(
        String::from("training"), training_data, training_size, feature_size, batch_size,
        Format::Binary, 573
        // Format::Text, 573
    );
    let mut testing_loader = DataLoader::from_scratch(
        String::from("testing"), testing_data, testing_size, feature_size, batch_size,
        Format::Binary, 573
        // Format::Text, 573
    );

    let args: Vec<String> = env::args().collect();
    if args[1] == "validate" {
        let mut reader = create_bufreader(&args[2]);
        let json = &read_k_lines(&mut reader, 1)[0];
        let model: Model = serde_json::from_str(&json).expect(
            &format!("Cannot parse the JSON description of the remote model. \
                        The JSON string is `{}`.", json)
        );
        let mut k = 0;
        while k * 10 <= model.len() {
            let k10 = if k == 0 { 1 } else { k * 10 };
            let model_subset = model[0..k10].to_vec();
            let scores = validate(&mut testing_loader, &model_subset, &eval_funcs);
            let output: Vec<String> = scores.into_iter().map(|x| x.to_string()).collect();
            info!("validate-only, {}, {}", k10, output.join(", "));
            k += 1;
        }
    } else if args[1] == "count" {
        let training_data = home_dir.clone() + "training.bin";
        let testing_data = home_dir.clone() + "testing.bin";
        let training_size = 50000000;
        let testing_size = 4627840;
        let mut training_loader = DataLoader::from_scratch(
            String::from("training"), training_data, training_size, feature_size, batch_size,
            Format::Binary, 573
            // Format::Text, 573
        );
        let mut testing_loader = DataLoader::from_scratch(
            String::from("testing"), testing_data, testing_size, feature_size, batch_size,
            Format::Binary, 573
        );
        let num_batches = training_loader.get_num_batches();
        let mut pos = 0.0;
        let mut neg = 0.0;
        (0..num_batches).for_each(|_| {
            training_loader.fetch_next_batch();
            let data = training_loader.get_curr_batch();
            let labels: Vec<f32> = data.iter()
                        .map(|d| get_symmetric_label(d))
                        .collect();
            labels.iter().for_each(|t| {
                if *t > 0.0 {
                    pos += 1.0;
                } else {
                    neg += 1.0;
                }
            });
            print!("{}, ", pos);
        });
        println!();

        let num_batches = testing_loader.get_num_batches();
        let mut pos = 0.0;
        let mut neg = 0.0;
        (0..num_batches).for_each(|_| {
            testing_loader.fetch_next_batch();
            let data = testing_loader.get_curr_batch();
            let labels: Vec<f32> = data.iter()
                        .map(|d| get_symmetric_label(d))
                        .collect();
            labels.iter().for_each(|t| {
                if *t > 0.0 {
                    pos += 1.0;
                } else {
                    neg += 1.0;
                }
            });
            print!("{}, ", pos);
        });
        println!();
    } else {
        assert_eq!(args.len(), 5);
        let local_name: String = args[1].clone();
        let range_1: usize = args[2].parse().unwrap();
        let range_2: usize = args[3].parse().unwrap();
        let num_iterations: usize = args[4].parse().unwrap();
        debug!("range, {}, {}", range_1, range_2);
        let mut boosting = Boosting::new(
            training_loader,
            testing_loader,
            range_1..range_2,
            max_sample_size,
            max_bin_size,
            sample_ratio,
            ess_threshold,
            default_rho_gamma,
            eval_funcs
        );
        if remote_ips.len() > 0 {
            boosting.enable_network(local_name, remote_ips, 8888);
        }
        boosting.training(
            num_iterations,
            max_trials_before_shrink,
            validate_interval
        );
    }
}

