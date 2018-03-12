#[macro_use] extern crate log;
extern crate env_logger;

mod bins;
mod commons;
mod data_loader; 
mod labeled_data;
mod learner;
mod tree;
mod validator;
mod boosting;


fn main() {
    env_logger::init();

    debug!("this is a debug {}", "message");
    error!("this is printed by default");
}
