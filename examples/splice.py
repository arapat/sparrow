#[macro_use] extern crate log;
extern crate rustboost;
extern crate env_logger;
extern crate serde_json;
extern crate time;

use std::env;
use std::fs;
use std::io::Read;
use std::io::Write;
use time::get_time;

use rustboost::commons::io::create_bufreader;
use rustboost::commons::io::read_k_lines;

use rustboost::boosting::Boosting;
use rustboost::buffer_loader::BufferLoader;
use rustboost::commons::LossFunc;
use rustboost::commons::Model;
use rustboost::config::Config;


fn main() {
    env_logger::Builder::from_default_env()
        .format(|buf, record| {
            let timestamp = get_time();
            let formatted_ts = format!("{}.{:09}", timestamp.sec, timestamp.nsec);
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

    let remote_ips = config.get_network();

    let training_data_filename = config.get_training_file();
    let testing_data_filename = config.get_testing_file();

    let training_size = config.get_training_size();
    let testing_size = config.get_testing_size();
    let feature_size = config.get_feature_size();
    let batch_size = config.get_batch_size();

    let eval_funcs: Vec<&LossFunc> = vec![];

    new(
        training_loader: BufferLoader, 
        range: Range<usize>, 
        max_sample_size: usize, 
        max_bin_size: usize, 
        default_gamma: f32
    )



    // -----
    let training_disk_reader = get_on_disk_reader(
        training_data_filename, true, training_size, feature_size, bytes_per_example);
    let training_reader = get_data_reader(training_size, batch_size, training_disk_reader);
    let training_scores_keeper = get_scores_keeper(training_size, batch_size);
    let training_loader = get_normal_loader(
        String::from("training"), training_size, training_reader, training_scores_keeper);

    let testing_disk_reader = get_on_disk_reader(
        testing_data_filename, true, testing_size, feature_size, bytes_per_example);
    let testing_reader = get_data_reader(testing_size, batch_size, testing_disk_reader);
    let testing_scores_keeper = get_scores_keeper(testing_size, batch_size);
    let mut testing_loader = get_normal_loader(
        String::from("testing"), testing_size, testing_reader, testing_scores_keeper);

    let args: Vec<String> = env::args().collect();
    if args[1] == "validate" {
        assert_eq!(args.len(), 4);
        assert!(args[2] == "reset" || args[2] == "noreset");
        testing_loader.get_data_reader().load_to_memory();
        let mut paths: Vec<String> = fs::read_dir(&args[3]).unwrap()
                                        .map(|p| format!("{}", p.unwrap().path().display()))
                                        .filter(|s| s.contains("model-"))
                                        .collect();
        info!("Collected {} model files.", paths.len());
        if args[2] == "reset" {
            info!("Running the reset version.");
        } else {
            info!("Running the noreset version.");
        }
        info!("Collected {} model files.", paths.len());
        paths.sort_by(|a, b| {
            let a_i: u32 = extract_num(a);
            let b_i: u32 = extract_num(b);
            a_i.cmp(&b_i)
        });
        let mut old_model = None;
        for path in paths {
            info!("Processing {}", path);
            let mut reader = create_bufreader(&path);
            let json = &read_k_lines(&mut reader, 1)[0];
            let (ts, model): (f32, Model) = serde_json::from_str(&json).expect(
                &format!("Cannot parse the JSON description of the remote model. \
                            The JSON string is `{}`.", json)
            );
            let output = validate(&mut testing_loader, &model, &eval_funcs);
            info!("validate-only, {}, {}, {}", model.len(), ts, output.join(", "));
            if args[2] == "reset" && !is_seq_model(&old_model, &model) {
                info!("now reset");
                testing_loader.reset_scores();
            }
            old_model = Some(model);
        }
    } else {
        assert_eq!(args.len(), 5);
        let local_name: String = args[1].clone();
        let range_1: usize = args[2].parse().unwrap();
        let range_2: usize = args[3].parse().unwrap();
        let num_iterations: usize = args[4].parse().unwrap();
        debug!("range, {}, {}", range_1, range_2);
        let mut boosting = Boosting::new(
            training_loader,
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