mod data_reader;
mod scores_in_mem;

use commons::get_weight;
use commons::Example;
use commons::Model;
use commons::PerformanceMonitor;

use self::data_reader::DataReader;
use self::data_reader::ActualReader;
use self::data_reader::data_on_disk::DataOnDisk;
use self::scores_in_mem::ScoresInMem;


pub fn get_on_disk_reader(filename: String, is_binary: bool, size: usize,
                          feature_size: usize, bytes_per_example: usize) -> ActualReader {
    ActualReader::OnDisk(
        DataOnDisk::new(filename, is_binary, size, feature_size, bytes_per_example)
    )
}


pub fn get_data_reader(size: usize, batch_size: usize, reader: ActualReader) -> DataReader {
    DataReader::new(size, batch_size, reader)
}


pub fn get_scores_keeper(size: usize, batch_size: usize) -> ScoresInMem {
    ScoresInMem::new(size, batch_size)
}


pub fn get_normal_loader(
        name: String, size: usize,
        data_reader: DataReader, scores_keeper: ScoresInMem) -> NormalLoader {
    NormalLoader::new(name, size, data_reader, scores_keeper)
}


#[derive(Debug)]
pub struct NormalLoader {
    name: String,
    size: usize,

    data_reader: DataReader,
    scores_keeper: ScoresInMem,

    sum_weights: f32,
    sum_weight_squared: f32,
    ess: Option<f32>,
}


impl NormalLoader {
    pub fn new(name: String, size: usize,
               data_reader: DataReader, scores_keeper: ScoresInMem) -> NormalLoader {
        // debug!(
        //     "new-data-loader, {}, {:?}, {}, {}, {}, {}, {}",
        //     filename, format, size, base_node, bytes_per_example
        // );

        NormalLoader {
            name: name,
            size: size,

            data_reader: data_reader,
            scores_keeper: scores_keeper,

            sum_weights: 0.0,
            sum_weight_squared: 0.0,
            ess: None,
        }
    }

    // Getter
    pub fn get_data_reader(&mut self) -> &mut DataReader {
        &mut self.data_reader
    }

    // Examples and scores
    pub fn get_num_batches(&self) -> usize {
        self.data_reader.get_num_batches()
    }

    pub fn get_curr_batch(&self) -> &Vec<Example> {
        self.data_reader.get_curr_batch()
    }

    pub fn get_relative_scores(&self) -> &[f32] {
        self.scores_keeper.get_relative_scores()
    }

    pub fn get_absolute_scores(&self) -> &[f32] {
        self.scores_keeper.get_absolute_scores()
    }

    pub fn fetch_next_batch(&mut self) {
        let (curr_loc, _) = self.data_reader.fetch_next_batch();
        self.scores_keeper.unset_scores_sync(curr_loc);
    }

    pub fn fetch_scores(&mut self, trees: &Model) {
        let scores_version = self.scores_keeper.get_scores_version();
        let updated_scores = {
            let curr_batch = self.get_curr_batch();
            let mut ret = vec![0.0; curr_batch.len()];
            for tree in trees[scores_version..trees.len()].iter() {
                tree.add_prediction_to_score(curr_batch, &mut ret);
            }
            ret
        };
        self.scores_keeper.update_scores(&updated_scores, trees.len());
    }

    pub fn reset_scores(&mut self) {
        self.scores_keeper.reset_scores();
    }

    // ESS and others
    pub fn get_ess(&self) -> Option<f32> {
        self.ess
    }

    fn update_stats_for_ess(&mut self) {
        let data = self.data_reader.fetch_next_batch();
        let mut sum_weights        = 0.0;
        let mut sum_weight_squared = 0.0;
        self.get_curr_batch()
            .iter()
            .zip(self.scores_keeper.get_relative_scores().iter())
            .for_each(|(data, score)| {
                let w = get_weight(data, *score);
                sum_weights += w;
                sum_weight_squared += w * w;
            });
        self.sum_weights        += sum_weights;
        self.sum_weight_squared += sum_weight_squared;
    }

    fn update_ess(&mut self) {
        let count = self.size;
        let ess = self.sum_weights.powi(2) / self.sum_weight_squared / (count as f32);
        debug!("loader-reset, {}, {}", self.name, ess);
        self.ess = Some(ess);
        self.sum_weights = 0.0;
        self.sum_weight_squared = 0.0;
    }

    fn report_timer(&mut self, timer: &mut PerformanceMonitor, timer_label: &str) {
        let (since_last_check, _, _, speed) = timer.get_performance();
        if since_last_check >= 300 {
            // debug!("{}, {}, {:?}, {}", timer_label, self.name, self.format, speed);
            timer.reset_last_check();
        }
    }
}