extern crate log;
extern crate env_logger;
extern crate time;

extern crate sparrow;

use std::env;
use std::io::Write;
use time::get_time;

use sparrow::testing;
use sparrow::training;


fn main() {
    let curr_time = get_time().sec;
    env_logger::Builder::from_default_env()
        .format(move |buf, record| {
            let timestamp = get_time();
            let formatted_ts = format!("{}.{}", timestamp.sec - curr_time, timestamp.nsec);
            writeln!(
                buf, "{}, {}, {}, {}",
                record.level(), formatted_ts, record.module_path().unwrap(), record.args()
            )
        })
        .init();

    let args: Vec<String> = env::args().collect();
    let usage_info = "Usage: ./sparrow [train|test] <config_file_path>";
    if args.len() != 3 {
        println!("{}", usage_info);
    } else if args[1] == "train" {
        training(&args[2]);
    } else if args[1] == "test" {
        testing(&args[2]);
    } else {
        println!("{}", usage_info);
    }
}