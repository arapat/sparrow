export INIT_SCRIPT="/mnt/rust-boost/scripts/init-c3_xlarge-ubuntu.sh"

cd /mnt

# Copy init script
scp -o StrictHostKeyChecking=no -i ~/jalafate-dropbox.pem $INIT_SCRIPT ubuntu@ec2-54-237-240-178.compute-1.amazonaws.com:~

# Execute init script
ssh -o StrictHostKeyChecking=no -i ~/jalafate-dropbox.pem ubuntu@ec2-54-237-240-178.compute-1.amazonaws.com sudo bash ~/init.sh

# Copy data
scp -o StrictHostKeyChecking=no -i ~/jalafate-dropbox.pem /mnt/*.bin ubuntu@ec2-54-237-240-178.compute-1.amazonaws.com:/mnt

# Clone repository
ssh -o StrictHostKeyChecking=no -i ~/jalafate-dropbox.pem ubuntu@ec2-54-237-240-178.compute-1.amazonaws.com git clone https://github.com/arapat/rust-boost.git /mnt/rust-boost

# Copy config file
scp -o StrictHostKeyChecking=no -i ~/jalafate-dropbox.pem /mnt/rust-boost/config.json ubuntu@ec2-54-237-240-178.compute-1.amazonaws.com:/mnt/rust-boost/config.json

# Install cargo
ssh -o StrictHostKeyChecking=no -i ./jalafate-dropbox.pem ubuntu@ec2-54-237-240-178.compute-1.amazonaws.com sudo apt-get update

ssh -o StrictHostKeyChecking=no -i ./jalafate-dropbox.pem ubuntu@ec2-54-237-240-178.compute-1.amazonaws.com sudo apt-get install -y cargo

# Build package
ssh -o StrictHostKeyChecking=no -i ~/jalafate-dropbox.pem ubuntu@ec2-54-237-240-178.compute-1.amazonaws.com "cd /mnt/rust-boost && cargo build --release"

