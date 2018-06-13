
/// Configuration of the RustBoost
#[derive(Serialize, Deserialize)]
pub struct Config {
    data_dir: String,
    network: Vec<String>
}

impl Config {
    /// Return the vector of the IPs of the neighbor workers
    pub fn get_network(&self) -> &Vec<String> {
        &self.network
    }

    /// Return the path to the directory that contains the training data
    pub fn get_data_dir(&self) -> &String {
        &self.data_dir
    }
}