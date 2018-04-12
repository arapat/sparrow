
use bufstream::BufStream;
use serde_json;

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
type PacketLoad = (String, u32, ModelScore);


pub fn start_network(
        name: String, init_remote_ips: &Vec<String>, port: u16,
        model_send: Sender<ModelScore>, model_recv: Receiver<ModelScore>) {
    let (ip_send, ip_recv): (Sender<SocketAddr>, Receiver<SocketAddr>) = mpsc::channel();
    start_sender(name.clone(), port, model_recv, ip_send.clone());
    start_receiver(name, port, model_send, ip_recv);

    // wait for other computers to be up and ready
    // TODO: waiting is not necessary if receive listener can handle
    // the connection refused exception
    sleep(Duration::from_secs(5));
    init_remote_ips.iter().for_each(|ip| {
        let socket_addr: SocketAddr =
            (ip.clone() + ":" + port.to_string().as_str()).parse().expect(
                &format!("Failed to parse initial remote IP `{}:{}`.", ip, port)
            );
        ip_send.send(socket_addr).expect(
            "Failed to send the initial remote IP to the receivers listener."
        );
    });
}

fn start_receiver(name: String, port: u16,
                  model_send: Sender<ModelScore>, remote_ip_recv: Receiver<SocketAddr>) {
    spawn(move|| {
        receivers_listener(name, port, model_send, remote_ip_recv);
    });
}

fn receivers_listener(name: String, port: u16,
                      model_send: Sender<ModelScore>, remote_ip_recv: Receiver<SocketAddr>) {
    info!("now entering receivers listener");
    let mut receivers = HashSet::new();
    loop {
        let mut remote_addr = remote_ip_recv.recv().expect(
            "Failed to unwrap the received remote IP."
        );
        remote_addr.set_port(port);
        if !receivers.contains(&remote_addr) {
            let name_clone = name.clone();
            let chan = model_send.clone();
            let addr = remote_addr.clone();
            receivers.insert(remote_addr.clone());
            spawn(move|| {
                let mut tcp_stream = None;
                while tcp_stream.is_none() {
                    tcp_stream = match TcpStream::connect(remote_addr) {
                        Ok(_tcp_stream) => Some(_tcp_stream),
                        Err(error) => {
                            info!("(retry in 2 secs) Error: {}.
                                  Failed to connect to remote address {}",
                                  error, remote_addr);
                            sleep(Duration::from_secs(2));
                            None
                        }
                    };
                }
                let stream = BufStream::new(tcp_stream.unwrap());
                receiver(name_clone, addr, stream, chan);
            });
        } else {
            info!("(Skipped) Receiver exists for {}", remote_addr);
        }
    }
}

fn receiver(name: String, remote_ip: SocketAddr,
            mut stream: BufStream<TcpStream>, chan: Sender<ModelScore>) {
    info!("Receiver started from {} to {}", name, remote_ip);
    let mut idx = 0;
    loop {
        let mut json = String::new();
        stream.read_line(&mut json).expect(
            "Cannot read the remote model from network."
        );
        if json.trim().len() != 0 {
            let (remote_name, remote_idx, model_score): PacketLoad =
                serde_json::from_str(&json).expect(
                    &format!("Cannot parse the JSON description of the remote model from {}. \
                              Message ID {}, JSON string is `{}`.", remote_ip, idx, json)
                );
            let model = model_score.0;
            let score = model_score.1;
            debug!("message-received, {}, {}, {}, {}, {}, {}, {}, \"{:?}\"",
                   name, idx, remote_name, remote_idx, remote_ip, score, json.len(), model);
            chan.send((model, score)).expect(
                "Failed to send the received model from the network to local channel."
            );
            idx += 1;
        } else {
            trace!("Received an empty message from {}, message ID {}", remote_ip, idx)
        }
    }
}

fn start_sender(name: String, port: u16,
                model_recv: Receiver<ModelScore>, remote_ip_send: Sender<SocketAddr>) {
    let streams: Vec<BufStream<TcpStream>> = vec![];
    let streams_arc = Arc::new(RwLock::new(streams));

    let arc_w = streams_arc.clone();
    let name_clone = name.clone();
    spawn(move|| {
        sender_listener(name_clone, port, arc_w, remote_ip_send);
    });

    spawn(move|| {
        sender(name, streams_arc, model_recv);
    });
}

fn sender_listener(
        name: String,
        port: u16,
        sender_streams: StreamLockVec,
        receiver_ips: Sender<SocketAddr>) {
    // Sender listener is responsible for:
    //     1. Add new incoming stream to sender (via streams RwLock)
    //     2. Send new incoming address to receiver so that it connects to the new machine
    info!("{} entering sender listener", name);
    let local_addr: SocketAddr =
        (String::from("0.0.0.0:") + port.to_string().as_str()).parse().expect(
            &format!("Cannot parse the port number `{}`.", port)
        );
    let listener = TcpListener::bind(local_addr)
        .expect(&format!("Failed to bind the listening port `{}`.", port));
    for stream in listener.incoming() {
        match stream {
            Err(_) => error!("Sender received an error connection."),
            Ok(stream) => {
                let remote_addr = stream.peer_addr().expect(
                    "Cannot unwrap the remote address from the incoming stream."
                );
                info!("Sender received a connection, {}, ->, {}",
                      remote_addr, stream.local_addr().expect(
                          "Cannot unwrap the local address from the incoming stream."
                      ));
                {
                    let mut lock_w = sender_streams.write().expect(
                        "Failed to obtain the lock for expanding sender_streams."
                    );
                    lock_w.push(BufStream::new(stream));
                }
                info!("Remote server {} will receive our model from now on.", remote_addr);
                receiver_ips.send(remote_addr.clone()).expect(
                    "Cannot send the received IP to the channel."
                );
                info!("Remote server {} will be subscribed soon (if not already).", remote_addr);
            }
        }
    }
}

fn sender(name: String, streams: StreamLockVec, chan: Receiver<ModelScore>) {
    info!("Sender has started.");

    let mut idx = 0;
    loop {
        let (model, score): ModelScore = chan.recv().expect(
            "Network module cannot receive the local model."
        );
        debug!("network-to-send-out, {}, {}, {}", name, idx, score);

        let packet_load: PacketLoad = (name.clone(), idx, (model, score));
        let json = serde_json::to_string(&packet_load).expect(
            "Local model cannot be serialized."
        );
        let num_computers = {
            let mut lock_r = streams.write().expect(
                "Failed to obtain the lock for writing to sender_streams."
            );
            lock_r.iter_mut().for_each(|stream| {
                stream.write_fmt(format_args!("{}\n", json)).expect(
                    "Cannot write into one of the streams."
                );
                stream.flush().expect(
                    "Cannot flush one of the streams."
                );
            });
            lock_r.len()
        };
        debug!("network-sent-out, {}, {}, {}, {}", name, idx, score, num_computers);
        idx += 1;
    }
}
