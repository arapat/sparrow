use std::fs::File;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;

// TODO: implement in-memory I/O buffer for both reading and writing


pub struct DiskBuffer {
    filename: String,
    block_size: usize,
    capacity: usize,
    size: usize,
    _file: File
}


impl DiskBuffer {
    pub fn new(filename: &str, block_size: usize, capacity: usize) -> DiskBuffer {
        let file = File::create(filename).expect(
            &format!("Cannot create the buffer file at {}", filename));
        DiskBuffer {
            filename: filename.to_string(),
            block_size: block_size,
            capacity: capacity,
            size: 0,
            _file: file
        }
    }

    pub fn write(&mut self, position: usize, data: &[u8]) {
        assert!(data.len() == self.block_size);
        if position <= self.size {
            let offset = position * self.block_size;
            self._file.seek(SeekFrom::Start(offset as u64)).expect(
                &format!("Cannot seek to the location {} while writing.", offset));
            self._file.write_all(data).unwrap();
        } else {
            self.size += 1;
            self._file.seek(SeekFrom::End(0)).expect(
                &format!("Cannot seek to the end of the file while writing."));
            self._file.write_all(data).unwrap();
        }
    }

    pub fn read(&mut self, position: usize) -> Vec<u8> {
        assert!(position < self.size);
        let offset = position * self.block_size;
        self._file.seek(SeekFrom::Start(offset as u64)).expect(
            &format!("Cannot seek to the location {} while reading.", offset));
        let mut block_buffer: Vec<u8> = vec![];
        self._file.read_exact(block_buffer.as_mut_slice()).unwrap();
        block_buffer
    }
}