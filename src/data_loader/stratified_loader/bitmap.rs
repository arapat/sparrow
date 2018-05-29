
pub struct BitMap {
    size: usize,
    is_free: Vec<i32>
}

impl BitMap {
    pub fn new(size: usize, all_full: bool) -> BitMap {
        let vec_size = (size + 31) / 32;
        let is_free = if all_full {
            vec![0; vec_size]
        } else {
            // all free
            vec![-1; vec_size]
        };
        BitMap {
            size: size,
            is_free: is_free
        }
    }

    pub fn get_first_free(&self) -> Option<usize> {
        let mut i = 0;
        let mut j = 0;
        while i * 32 < self.size {
            let k: i64 = self.is_free[i] as i64;
            if k == 0 {
                i += 1;
            } else {
                j = k & -k;
                break;
            }
        }
        let ret = i * 32 + BitMap::log(j);
        if j == 0 || ret >= self.size {
            None
        } else {
            Some(ret)
        }
    }

    pub fn mark_free(&mut self, position: usize) {
        assert!(position < self.size);
        let div = position / 32;
        let res = position % 32;
        self.is_free[div] = self.is_free[div] | (1 << res);
    }

    pub fn mark_filled(&mut self, position: usize) {
        assert!(position < self.size);
        let div = position / 32;
        let res = position % 32;
        self.is_free[div] = self.is_free[div] & !(1 << res);
    }

    fn log(t: i64) -> usize {
        let mut left = 0;
        let mut right = 32 + 1;
        while left + 1 < right {
            let mid = (left + right) >> 1;
            if t >> mid == 0{
                right = mid;
            } else {
                left = mid;
            }
        }
        left as usize
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_full() {
        let mut bitmap = BitMap::new(200, true);
        assert_eq!(None, bitmap.get_first_free());
        bitmap.mark_filled(105);
        assert_eq!(None, bitmap.get_first_free());
        bitmap.mark_free(105);
        assert_eq!(Some(105), bitmap.get_first_free());
        bitmap.mark_filled(105);
        assert_eq!(None, bitmap.get_first_free());
    }

    #[test]
    fn test_all_empty() {
        let mut bitmap = BitMap::new(200, false);
        assert_eq!(Some(0), bitmap.get_first_free());
        bitmap.mark_filled(0);
        assert_eq!(Some(1), bitmap.get_first_free());
        bitmap.mark_filled(1);
        assert_eq!(Some(2), bitmap.get_first_free());
        bitmap.mark_free(0);
        assert_eq!(Some(0), bitmap.get_first_free());
    }
}