echo "Launching on this computer"
cd /mnt/rust-boost && RUST_LOG=DEBUG nohup cargo run --release 0 164 300 2> run-network.log &

echo "Launching on computer 1"
ssh -n -o StrictHostKeyChecking=no -i /mnt/jalafate-dropbox.pem ubuntu@ec2-54-237-240-178.compute-1.amazonaws.com "cd /mnt/rust-boost; RUST_LOG=DEBUG nohup cargo run --release 164 264 500 2> run-network.log 1>&2 < /dev/null &"
echo "Launching on computer 2"
ssh -n -o StrictHostKeyChecking=no -i /mnt/jalafate-dropbox.pem ubuntu@ec2-54-236-141-208.compute-1.amazonaws.com "cd /mnt/rust-boost; RUST_LOG=DEBUG nohup cargo run --release 264 364 500 2> run-network.log 1>&2 < /dev/null &"
echo "Launching on computer 3"
ssh -n -o StrictHostKeyChecking=no -i /mnt/jalafate-dropbox.pem ubuntu@ec2-34-235-135-227.compute-1.amazonaws.com "cd /mnt/rust-boost; RUST_LOG=DEBUG nohup cargo run --release 364 464 500 2> run-network.log 1>&2 < /dev/null &"
echo "Launching on computer 4"
ssh -n -o StrictHostKeyChecking=no -i /mnt/jalafate-dropbox.pem ubuntu@ec2-54-173-36-174.compute-1.amazonaws.com  "cd /mnt/rust-boost; RUST_LOG=DEBUG nohup cargo run --release 464 564 500 2> run-network.log 1>&2 < /dev/null &"

