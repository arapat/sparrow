extern crate bufstream;
extern crate serde_json;

use self::bufstream::BufStream;

use std::io::Write;
use std::net::TcpStream;
use std::thread::spawn;
use std::io::BufRead;
use std::sync::mpsc::Sender;
use std::sync::mpsc::Receiver;

use commons::ModelScore;


fn receiver(remote_ip: String, mut stream: BufStream<TcpStream>, chan: Sender<ModelScore>) {
    debug!("Receiver for `{}` has started.", remote_ip);
    let mut idx = 0;
    loop {
        let mut json = String::new();
        stream.read_line(&mut json).unwrap();
        if json.trim().len() != 0 {
            debug!("Message {}: received `{}` from `{}`. Message length {}.",
                   idx, json, remote_ip, json.len());
            let (model, score): ModelScore = serde_json::from_str(&json).unwrap();
            debug!("The score of the Model {} from `{}` is {}.", idx, remote_ip, score);
            chan.send((model, score)).unwrap();
        } else {
            debug!("Received an empty message from `{}`.", remote_ip)
        }
        idx += 1;
    }
}

fn sender(remote_ips: Vec<String>, port: String, chan: Receiver<ModelScore>) {
    debug!("Sender has started.");

    let num_computers = remote_ips.len();
    let mut streams: Vec<BufStream<TcpStream>> = remote_ips.into_iter().map(|remote_ip| {
        let addr = remote_ip + ":" + port.as_str();
        BufStream::new(TcpStream::connect(addr).unwrap())
    }).collect();
    info!("Sender has created {} connections to the remote computers.", num_computers);

    let mut idx = 0;
    loop {
        let (model, score): ModelScore = chan.recv().unwrap();
        debug!("Local model {}, score {}, is received by the sender.", idx, score);

        let json = serde_json::to_string(&(model, score)).unwrap();
        streams.iter_mut().for_each(|stream| {
            stream.write_fmt(format_args!("{}\n", json)).unwrap();
            stream.flush().unwrap();
        });
        info!("Local model {}, score {}, is sent out to {} computers.", idx, score, num_computers);
        idx += 1;
    }
}

pub fn start_receivers(remote_ips: &Vec<String>, port: &str, send: Sender<ModelScore>) {
    for remote_ip in remote_ips {
        let chan = send.clone();
        let remote_ip = remote_ip.clone();
        let addr = remote_ip.clone() + ":" + port;
        spawn(move|| {
            let stream = BufStream::new(
                TcpStream::connect(addr).unwrap()
            );
            receiver(remote_ip, stream, chan);
        });
    }
}

pub fn start_sender(remote_ips: &Vec<String>, port: &str, receive: Receiver<ModelScore>) {
    let port = String::from(port);
    let remote_ips = remote_ips.clone();
    spawn(move|| {
        sender(remote_ips, port, receive);
    });
}
