use std::thread::spawn;
use std::thread::sleep;
use std::time::Duration;
use crossbeam_channel as channel;


pub struct Sender<T> {
    sender: channel::Sender<T>,
}


impl<T> Sender<T> {
    pub fn new(sender: channel::Sender<T>) -> Sender<T> {
        Sender {
            sender: sender,
        }
    }

    pub fn send(&self, t: T) {
        self.sender.send(t)
    }
}


impl<T> Clone for Sender<T> {
    fn clone(&self) -> Sender<T> {
        Sender::new(self.sender.clone())
    }
}


pub struct Receiver<T> {
    receiver: channel::Receiver<T>,
}


impl<T> Receiver<T> {
    pub fn new(receiver: channel::Receiver<T>) -> Receiver<T> {
        Receiver {
            receiver: receiver,
        }
    }

    pub fn recv(&self) -> Option<T> {
        self.receiver.recv()
    }

    pub fn try_recv(&self) -> Option<T> {
        self.receiver.try_recv()
    }
}


impl<T> Clone for Receiver<T> {
    fn clone(&self) -> Receiver<T> {
        Receiver::new(self.receiver.clone())
    }
}


pub fn bounded<T>(size: usize, name: &str) -> (Sender<T>, Receiver<T>)
where T: 'static + Send {
    let (sender, receiver) = channel::bounded(size);
    let name = String::from(name);
    {
        let sender = sender.clone();
        spawn(move || {
            loop {
                debug!("channel status, {}, {}, {}", name, sender.len(), size);
                sleep(Duration::from_millis(5000));
            }
        });
    }
    (Sender::new(sender), Receiver::new(receiver))
}
