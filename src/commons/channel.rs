use std::thread::spawn;
use std::thread::sleep;
use std::collections::HashMap;
use std::time::Duration;
use crossbeam_channel;


lazy_static! {
    static ref CHANNEL_MONITOR: ChannelMonitor = ChannelMonitor::new();
}


enum StatType {
    Send,
    BlockedSend,
    Recv,
    BlockedRecv,
}


pub struct ChannelMonitor {
    stats_sender: crossbeam_channel::Sender<(String, StatType, u32)>,
    #[allow(dead_code)] stats_receiver: crossbeam_channel::Receiver<(String, StatType, u32)>,
}


impl ChannelMonitor {
    fn new() -> ChannelMonitor {
        let (sender, receiver) = crossbeam_channel::unbounded();
        {
            let receiver = receiver.clone();
            spawn(move || {
                let mut stats: HashMap<String, ((u32, u32), (u32, u32))> = HashMap::new();
                loop {
                    sleep(Duration::from_millis(2000));
                    while let Some((name, stat_type, k)) = receiver.try_recv() {
                        let entry = stats.entry(name).or_insert(((0, 0), (0, 0)));
                        match stat_type {
                            StatType::Send        => { ((*entry).0).0 += k; },
                            StatType::BlockedSend => { ((*entry).0).1 += k; },
                            StatType::Recv        => { ((*entry).1).0 += k; },
                            StatType::BlockedRecv => { ((*entry).1).1 += k; },
                        }
                    }
                    for (name, val) in stats.iter() {
                        let ((send, blocked_send), (recv, blocked_recv)) = val;
                        debug!("channel status, {}, {}, {}, {}, {}",
                               name, send, blocked_send, recv, blocked_recv);
                    }
                    stats.clear();
                }
            });
        }
        ChannelMonitor {
            stats_sender: sender,
            stats_receiver: receiver,
        }
    }
}


pub struct Sender<T> {
    name: String,
    sender: crossbeam_channel::Sender<T>,
    stats_sender: crossbeam_channel::Sender<(String, StatType, u32)>,
}


impl<T> Sender<T> {
    fn new(
        name: String,
        sender: crossbeam_channel::Sender<T>,
    ) -> Sender<T> {
        Sender {
            name: name,
            sender: sender,
            stats_sender: CHANNEL_MONITOR.stats_sender.clone(),
        }
    }

    pub fn send(&self, t: T) {
        self.stats_sender.send((self.name.clone(), StatType::Send, 1));
        select! {
            send(self.sender, t) => (),
            default => {
                self.stats_sender.send((self.name.clone(), StatType::BlockedSend, 1));
                self.sender.send(t);
            }
        }
    }
}


impl<T> Clone for Sender<T> {
    fn clone(&self) -> Sender<T> {
        Sender {
            name: self.name.clone(),
            sender: self.sender.clone(),
            stats_sender: self.stats_sender.clone(),
        }
    }
}


pub struct Receiver<T> {
    name: String,
    receiver: crossbeam_channel::Receiver<T>,
    stats_sender: crossbeam_channel::Sender<(String, StatType, u32)>,
}


impl<T> Receiver<T> {
    fn new(
        name: String,
        receiver: crossbeam_channel::Receiver<T>,
    ) -> Receiver<T> {
        Receiver {
            name: name,
            receiver: receiver,
            stats_sender: CHANNEL_MONITOR.stats_sender.clone(),
        }
    }

    pub fn recv(&self) -> Option<T> {
        self.stats_sender.send((self.name.clone(), StatType::Recv, 1));
        select! {
            recv(self.receiver, t) => t,
            default => {
                self.stats_sender.send((self.name.clone(), StatType::BlockedRecv, 1));
                self.receiver.recv()
            }
        }
    }

    pub fn try_recv(&self) -> Option<T> {
        self.stats_sender.send((self.name.clone(), StatType::Recv, 1));
        select! {
            recv(self.receiver, t) => t,
            default => {
                self.stats_sender.send((self.name.clone(), StatType::BlockedRecv, 1));
                None
            }
        }
    }
}


impl<T> Clone for Receiver<T> {
    fn clone(&self) -> Receiver<T> {
        Receiver {
            name: self.name.clone(),
            receiver: self.receiver.clone(),
            stats_sender: self.stats_sender.clone(),
        }
    }
}


pub fn bounded<T>(
    size: usize,
    name: &str,
) -> (Sender<T>, Receiver<T>) {
    let (sender, receiver) = crossbeam_channel::bounded(size);
    let name = String::from(name);
    (Sender::new(name.clone(), sender), Receiver::new(name.clone(), receiver))
}
