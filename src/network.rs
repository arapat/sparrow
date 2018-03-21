extern crate bufstream;
extern crate serde_json;

use self::bufstream::BufStream;

use std::collections::HashSet;
use std::io::Write;
use std::io::BufRead;
use std::net::TcpListener;
use std::net::TcpStream;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::mpsc;
use std::sync::mpsc::Sender;
use std::sync::mpsc::Receiver;
use std::thread::spawn;
use std::thread::sleep;
use std::time::Duration;

use commons::ModelScore;

type StreamLockVec = Arc<RwLock<Vec<BufStream<TcpStream>>>>;


pub fn start_network(
        init_remote_ips: &Vec<String>, port: u16,
        model_send: Sender<ModelScore>, model_recv: Receiver<ModelScore>) {
    let (ip_send, ip_recv): (Sender<SocketAddr>, Receiver<SocketAddr>) = mpsc::channel();
    start_sender(port, model_recv, ip_send.clone());
    start_receiver(port, model_send, ip_recv);

    // wait for other computers to be up and ready
    // TODO: waiting is not necessary if receive listener can handle
    // the connection refused exception
    sleep(Duration::from_secs(5));
    init_remote_ips.iter().for_each(|ip| {
        let socket_addr: SocketAddr =
            (ip.clone() + ":" + port.to_string().as_str()).parse().unwrap();
        ip_send.send(socket_addr).unwrap();
    });
}

fn start_receiver(port: u16, model_send: Sender<ModelScore>, remote_ip_recv: Receiver<SocketAddr>) {
    spawn(move|| {
        receivers_listener(port, model_send, remote_ip_recv);
    });
}

fn receivers_listener(port: u16, model_send: Sender<ModelScore>,
                      remote_ip_recv: Receiver<SocketAddr>) {
    debug!("now entering receivers listener");
    let mut receivers = HashSet::new();
    loop {
        let mut remote_addr = remote_ip_recv.recv().unwrap();
        remote_addr.set_port(port);
        if !receivers.contains(&remote_addr) {
            let chan = model_send.clone();
            let addr = remote_addr.clone();
            receivers.insert(remote_addr.clone());
            spawn(move|| {
                let stream = BufStream::new(
                    TcpStream::connect(remote_addr).unwrap()
                );
                receiver(addr, stream, chan);
            });
        } else {
            info!("Receiver for `{}` exists, skipped..", remote_addr);
        }
    }
}

fn receiver(remote_ip: SocketAddr, mut stream: BufStream<TcpStream>, chan: Sender<ModelScore>) {
    info!("Receiver for `{}` has started.", remote_ip);
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

fn start_sender(port: u16, model_recv: Receiver<ModelScore>, remote_ip_send: Sender<SocketAddr>) {
    let streams: Vec<BufStream<TcpStream>> = vec![];
    let streams_arc = Arc::new(RwLock::new(streams));

    let arc_w = streams_arc.clone();
    spawn(move|| {
        sender_listener(port, arc_w, remote_ip_send);
    });

    spawn(move|| {
        sender(streams_arc, model_recv);
    });
}

fn sender_listener(
        port: u16,
        sender_streams: StreamLockVec,
        receiver_ips: Sender<SocketAddr>) {
    // Sender listener is responsible for:
    //     1. Add new incoming stream to sender (via streams RwLock)
    //     2. Send new incoming address to receiver so that it connects to the new machine
    debug!("now entering sender listener");
    let local_addr: SocketAddr =
        (String::from("0.0.0.0:") + port.to_string().as_str()).parse().unwrap();
    let listener = TcpListener::bind(local_addr).unwrap();
    for stream in listener.incoming() {
        match stream {
            Err(_) => error!("Sender received an error connection."),
            Ok(mut stream) => {
                let remote_addr = stream.peer_addr().unwrap();
                info!("Sender received a connection from {} to {}",
                      remote_addr, stream.local_addr().unwrap());
                {
                    let mut lock_w = sender_streams.write().unwrap();
                    lock_w.push(BufStream::new(stream));
                }
                debug!("Remote server {} will receive our model from now on.", remote_addr);
                receiver_ips.send(remote_addr.clone()).unwrap();
                debug!("Remote server {} will be subscribed soon.", remote_addr);
            }
        }
    }
}

fn sender(streams: StreamLockVec, chan: Receiver<ModelScore>) {
    debug!("Sender has started.");

    let mut idx = 0;
    loop {
        let (model, score): ModelScore = chan.recv().unwrap();
        debug!("Local model {}, score {}, is received by the sender.", idx, score);

        let json = serde_json::to_string(&(model, score)).unwrap();
        let num_computers = {
            let mut lock_r = streams.write().unwrap();
            lock_r.iter_mut().for_each(|stream| {
                stream.write_fmt(format_args!("{}\n", json)).unwrap();
                stream.flush().unwrap();
            });
            lock_r.len()
        };
        info!("Local model {}, score {}, is sent out to {} computers.", idx, score, num_computers);
        idx += 1;
    }
}
