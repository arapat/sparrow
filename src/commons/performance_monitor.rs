use std::fmt;

use time::PreciseTime;


// Performance monitoring
#[derive(PartialEq)]
enum PerformanceMonitorStatus {
    PAUSE,
    RUNNING
}

pub struct PerformanceMonitor {
    start_time: PreciseTime,
    last_check: PreciseTime,
    status: PerformanceMonitorStatus,
    duration: i64,
    duration_adjust: f32,
    counter: usize
}

impl fmt::Debug for PerformanceMonitor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PerformanceMonitor")
    }
}

impl PerformanceMonitor {
    pub fn new() -> PerformanceMonitor {
        PerformanceMonitor {
            start_time: PreciseTime::now(),
            last_check: PreciseTime::now(),
            status: PerformanceMonitorStatus::PAUSE,
            duration: 0,
            duration_adjust: 0.0,
            counter: 0,
        }
    }

    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.duration = 0;
        self.counter = 0;
        self.status = PerformanceMonitorStatus::PAUSE;
        self.last_check = PreciseTime::now();
    }

    pub fn start(&mut self) {
        self.resume();
    }

    pub fn resume(&mut self) {
        assert!(self.status == PerformanceMonitorStatus::PAUSE);
        self.start_time = PreciseTime::now();
        self.status = PerformanceMonitorStatus::RUNNING;
    }

    pub fn update(&mut self, count: usize) {
        assert!(self.status == PerformanceMonitorStatus::RUNNING);
        self.counter += count;
    }

    pub fn pause(&mut self) {
        assert!(self.status == PerformanceMonitorStatus::RUNNING);
        self.duration += self.start_time.to(PreciseTime::now()).num_microseconds().unwrap();
        self.status = PerformanceMonitorStatus::PAUSE;
    }

    pub fn write_log(&mut self, name: &str) -> bool {
        let (since_last_check, count, duration, speed) = self.get_performance();
        if since_last_check >= 20 {
            debug!("{}-perf-mon, {:.2}, {}, {:.2}", name, duration, count, speed);
            self.last_check = PreciseTime::now();
            true
        } else {
            false
        }
    }

    pub fn get_duration(&self) -> f32 {
        let microseconds = self.duration +
            if self.status == PerformanceMonitorStatus::RUNNING {
                self.start_time.to(PreciseTime::now()).num_microseconds().unwrap()
            } else {
                0
            };
        1e-6 * microseconds as f32 + self.duration_adjust
    }

    #[allow(dead_code)]
    pub fn get_counts(&self) -> usize {
        self.counter
    }

    fn get_performance(&mut self) -> (i64, usize, f32, f32) {
        let since_last_check = self.last_check.to(PreciseTime::now()).num_seconds();
        let duration = self.get_duration();
        (since_last_check, self.counter, duration, (self.counter as f32) / duration)
    }
}