
#[derive(Serialize, Deserialize)]
pub struct Config {
    data_dir: String,
    network: Vec<String>
}

impl Config {
    pub fn get_network(&self) -> &Vec<String> {
        &self.network
    }

    pub fn get_data_dir(&self) -> &String {
        &self.data_dir
    }
}