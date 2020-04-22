use std::fs::File;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;

use bincode::serialize;
use std::fs::remove_file;
use super::bitmap::BitMap;

// TODO: implement in-memory I/O buffer for both reading and writing


#[derive(Serialize, Deserialize)]
pub struct DiskBuffer {
    bitmap: BitMap,
    block_size: usize,
    capacity: usize,
    size: usize,
    #[serde(skip)] file: Option<File>,
    filename: String,
}


impl DiskBuffer {
    pub fn new(filename: &str, block_size: usize, capacity: usize) -> DiskBuffer {
        let mut disk_buffer = DiskBuffer {
            bitmap: BitMap::new(capacity.clone(), false),
            block_size: block_size,
            capacity: capacity,
            size: 0,
            file: None,
            filename: String::from(filename),
        };
        disk_buffer.init_file();
        disk_buffer
    }

    pub fn write(&mut self, data: &[u8]) -> usize {
        assert!(data.len() == self.block_size);
        let position = {
            let idx = self.bitmap.get_first_free().expect(
                "No free slot available."
            );
            self.bitmap.mark_filled(idx);
            idx
        };
        assert!(position <= self.size);
        if position >= self.size {
            self.size += 1;
        }
        assert!(self.size <= self.capacity);
        self.write_at(position, data)
    }

    pub fn write_at(&mut self, position: usize, data: &[u8]) -> usize {
        let offset = position * self.block_size;
        let file = self.file.as_mut().unwrap();
        file.seek(SeekFrom::Start(offset as u64)).expect(
            &format!("Cannot seek to the location {} while writing.", offset));
        file.write_all(data).unwrap();
        file.flush().unwrap();
        position
    }

    pub fn read(&mut self, position: usize) -> Vec<u8> {
        assert!(position < self.size);
        let ret = self.read_at(position);
        self.bitmap.mark_free(position);
        ret
    }

    pub fn read_at(&mut self, position: usize) -> Vec<u8> {
        let offset = position * self.block_size;
        let file = self.file.as_mut().unwrap();
        file.seek(SeekFrom::Start(offset as u64)).expect(
            &format!("Cannot seek to the location {} while reading.", offset));
        let mut block_buffer: Vec<u8> = vec![0; self.block_size];
        file.read_exact(block_buffer.as_mut_slice()).expect(
            &format!("Read from disk failed. Disk buffer size is `{}`. Position to read is `{}`.",
                     self.size, position)
        );
        block_buffer
    }

    #[allow(dead_code)]
    pub fn serialize(&self) -> Vec<u8> {
        serialize(self).unwrap()
    }

    pub fn init_file(&mut self) {
        self.file = Some(
            OpenOptions::new()
             .read(true)
             .write(true)
             .create(true)
             .open(self.filename.clone()).expect(
                 &format!("Cannot create the buffer file at {}", self.filename))
        );
    }

    pub fn get_all_filled(&self) -> Vec<usize> {
        self.bitmap.get_all_filled()
    }
}

impl Drop for DiskBuffer {
    fn drop(&mut self) {
        remove_file(&self.filename).unwrap();
    }
}


// #[cfg(test)]
// mod tests {
//     use bincode::serialize;
//     use bincode::deserialize;
// 
//     use labeled_data::LabeledData;
//     use commons::ExampleWithScore;
//     use super::super::get_disk_buffer;
//     use TFeature;
// 
// 
//     #[test]
//     fn test_disk_buffer_normal_write() {
//         let filename = "unittest-diskbuffer1.bin";
//         let mut disk_buffer = get_disk_buffer(filename, 3, 50, 10);
//         let example = get_example(vec![1, 2, 3]);
//         let data = serialize(&vec![example; 10]).unwrap();
//         for _ in 0..5 {
//             disk_buffer.write(&data);
//         }
//     }
// 
//     #[test]
//     fn test_disk_buffer_rw_once() {
//         let filename = "unittest-diskbuffer2.bin";
//         let mut disk_buffer = get_disk_buffer(filename, 3, 50, 10);
//         let example = get_example(vec![4, 5, 6]);
//         let examples = vec![example; 10];
//         let data = serialize(&examples).unwrap();
//         let index = disk_buffer.write(&data);
//         let retrieve = disk_buffer.read(index);
//         assert_eq!(data, retrieve);
//         let examples_des: Vec<ExampleWithScore> = deserialize(&retrieve).unwrap();
//         assert_eq!(examples, examples_des);
//     }
// 
//     #[test]
//     fn test_disk_buffer_rw_one_by_one() {
//         let filename = "unittest-diskbuffer3.bin";
//         let mut disk_buffer = get_disk_buffer(filename, 3, 50, 10);
//         for i in 0..5 {
//             let example = get_example(vec![1, 2, i as TFeature]);
//             let examples = vec![example; 10];
//             let data = serialize(&examples).unwrap();
//             let index = disk_buffer.write(&data);
//             assert_eq!(index, 0);
//             let retrieve = disk_buffer.read(index);
//             assert_eq!(data, retrieve);
//             let examples_des: Vec<ExampleWithScore> = deserialize(&retrieve).unwrap();
//             assert_eq!(examples, examples_des);
//         }
//     }
// 
//     #[test]
//     fn test_disk_buffer_rw_seq() {
//         let filename = "unittest-diskbuffer4.bin";
//         let mut disk_buffer = get_disk_buffer(filename, 3, 50, 10);
//         let mut inputs = vec![];
//         for i in 0..5 {
//             let example = get_example(vec![1, 2, i as TFeature]);
//             let examples = vec![example; 10];
//             let data = serialize(&examples).unwrap();
//             let index = disk_buffer.write(&data);
//             inputs.push((data, examples));
//             assert_eq!(index, i as usize);
//         }
//         for i in 0..5 {
//             let retrieve = disk_buffer.read(i);
//             assert_eq!(inputs[i].0, retrieve);
//             let examples_des: Vec<ExampleWithScore> = deserialize(&retrieve).unwrap();
//             assert_eq!(inputs[i].1, examples_des);
//         }
//     }
// 
//     #[test]
//     #[should_panic]
//     fn test_disk_buffer_read_panic() {
//         let filename = "unittest-diskbuffer5.bin";
//         let mut disk_buffer = get_disk_buffer(filename, 3, 10, 10);
//         disk_buffer.read(0);
//     }
// 
//     #[test]
//     #[should_panic]
//     fn test_disk_buffer_write_disk_full_panic() {
//         let filename = "unittest-diskbuffer6.bin";
//         let mut disk_buffer = get_disk_buffer(filename, 3, 50, 10);
//         let example = get_example(vec![1, 2, 3]);
//         let data = serialize(&vec![example; 10]).unwrap();
//         for _ in 0..6 {
//             disk_buffer.write(&data);
//         }
//     }
// 
//     #[test]
//     #[should_panic]
//     fn test_disk_buffer_write_feature_size_mismatch_panic() {
//         let filename = "unittest-diskbuffer7.bin";
//         let mut disk_buffer = get_disk_buffer(filename, 3, 50, 10);
//         let example = get_example(vec![1, 2, 3, 4]);
//         let data = serialize(&vec![example; 10]).unwrap();
//         disk_buffer.write(&data);
//     }
// 
//     fn get_example(features: Vec<TFeature>) -> ExampleWithScore {
//         let label: i8 = -1;
//         let example = LabeledData::new(features, label);
//         (example, (1.0, 0))
//     }
// }