mod data_reader;
mod normal_loader;
mod sample_cons;
mod scores_in_mem;

use self::data_reader::DataReader;
use self::data_reader::ActualReader;
use self::data_reader::data_on_disk::DataOnDisk;
pub use self::normal_loader::NormalLoader;
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