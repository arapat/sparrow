extern crate log;
extern crate env_logger;
extern crate time;

extern crate sparrow;

use std::io::Write;
use time::get_time;

use sparrow::run_rust_boost;


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

    let config_file = String::from("./examples/config_bathymetry.yaml");
    run_rust_boost(config_file);
}