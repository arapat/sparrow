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
mod data_loader; 
mod labeled_data;
mod learner;
mod tree;
mod validator;
mod boosting;
mod network;

use std::env;
use validator::get_adaboost_loss;
use validator::get_auprc;
use data_loader::Format;
use data_loader::DataLoader;
use boosting::Boosting;
use commons::LossFunc;


fn main() {
    env_logger::init();

    // read from text
    // let home_dir = std::env::home_dir().unwrap().display().to_string() +
    //                "/Downloads/splice/";
    // let training_data = home_dir.clone() + "training-shuffled.txt";
    // let testing_data =  home_dir.clone() + "testing-shuffled.txt";

    // read from bin
    let home_dir = String::from("./bin-data/");
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
    let batch_size = 1000;

    let max_sample_size = 10000;
    let max_bin_size = 2;
    let sample_ratio = 0.05;
    let ess_threshold = 0.5;
    let default_rho_gamma = 0.25;
    let eval_funcs: Vec<&LossFunc> = vec![
        &get_adaboost_loss,
        &get_auprc
    ];
    let num_iterations = 0;
    let max_trials_before_shrink = 1000000;
    let validate_interval = 10;

    let training_loader = DataLoader::from_scratch(
        String::from("training"), training_data, training_size, feature_size, batch_size,
        Format::Binary, 573
        // Format::Text, 573
    );
    let testing_loader = DataLoader::from_scratch(
        String::from("testing"), testing_data, testing_size, feature_size, batch_size,
        Format::Binary, 573
        // Format::Text, 573
    );

    let remote_ips = vec![
        String::from("18.232.106.228"),
        String::from("34.207.76.228"),
        String::from("54.224.67.100"),
        String::from("34.229.241.77"),
        String::from("34.235.119.198")
    ];

    let args: Vec<String> = env::args().collect();
    let range_1: usize = args[1].parse().unwrap();
    let range_2: usize = args[2].parse().unwrap();
    debug!("Range: {}..{}", range_1, range_2);
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
    boosting.enable_network(&remote_ips, 8888);
    boosting.training(
        num_iterations,
        max_trials_before_shrink,
        validate_interval
    );
}
