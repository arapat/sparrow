
#[derive(Serialize, Deserialize)]
pub struct Config {
    network: Vec<String>
}

impl Config {
    pub fn network(&self) -> &Vec<String> {
        &self.network
    }
}