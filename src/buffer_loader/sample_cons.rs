use commons::Example;
use commons::get_symmetric_label;
use commons::is_positive;

use buffer_loader::normal_loader::NormalLoader;
use buffer_loader::data_reader::ActualReader;
use buffer_loader::data_reader::DataReader;
use buffer_loader::data_reader::data_in_mem::DataInMem;
use buffer_loader::scores_in_mem::ScoresInMem;


#[derive(Debug)]
pub struct SampleCons {
    size: usize,
    capacity: usize,

    data: DataInMem,
    scores: Vec<f32>,

    num_positive: usize,
    num_negative: usize
}


impl SampleCons {
    pub fn new(capacity: usize) -> SampleCons {
        let data = DataInMem::new(capacity);
        let scores = Vec::with_capacity(capacity);
        SampleCons {
            size: 0,
            capacity: capacity,

            data: data,
            scores: scores,

            num_positive: 0,
            num_negative: 0
        }
    }

    pub fn construct(mut self, batch_size: usize, base_node: usize) -> NormalLoader {
        // this function is intentionally destructing the struct
        debug!("sample-construct, {}, {}, {}", self.size, self.num_positive, self.num_negative);
        let mut scores_keeper = ScoresInMem::new(self.size, batch_size);
        self.scores.shrink_to_fit();
        scores_keeper.set_base_scores(base_node, self.scores);
        let data_reader = DataReader::new(self.size, batch_size, ActualReader::InMem(self.data));
        NormalLoader::new(String::from("sample"), self.size, data_reader, scores_keeper)
    }

    pub fn append_data(&mut self, data: &Example, score: f32) -> bool {
        self.data.append(data);
        self.scores.push(score);
        self.size += 1;
        if is_positive(&get_symmetric_label(data)) {
            self.num_positive += 1;
        } else {
            self.num_negative += 1;
        }
        self.size >= self.capacity
    }
}