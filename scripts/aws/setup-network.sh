nodes=(
ec2-54-172-121-2.compute-1.amazonaws.com
ec2-54-237-240-178.compute-1.amazonaws.com
ec2-54-236-141-208.compute-1.amazonaws.com
ec2-34-235-135-227.compute-1.amazonaws.com
ec2-54-173-36-174.compute-1.amazonaws.com
)

export INIT_SCRIPT="/mnt/rust-boost/scripts/init-c3_xlarge-ubuntu.sh"
export IDENT_FILE=""
export GIT_REPO="https://github.com/arapat/rust-boost.git"
export GIT_BRANCH="~/jalafate-dropbox.pem"

for i in "${!nodes[@]}";
do
    # Copy init script
    scp -o StrictHostKeyChecking=no -i $IDENT_FILE $INIT_SCRIPT ubuntu@$url:~

    # Execute init script
    ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url sudo bash ~/init.sh

    # Copy data
    scp -o StrictHostKeyChecking=no -i $IDENT_FILE /mnt/*.bin ubuntu@$url:/mnt

    # Clone repository
    ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url git clone $GIT_REPO /mnt/rust-boost

    # Install cargo
    ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url sudo apt-get update
    ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url sudo apt-get install -y cargo

    # Build package
    ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url "
        cd /mnt/rust-boost && git checkout -- . && git fetch --all &&
        git checkout $GIT_BRANCH && git pull && cargo build --release"

    # Copy config file
    scp -o StrictHostKeyChecking=no -i $IDENT_FILE /mnt/rust-boost/config.json ubuntu@$url:/mnt/rust-boost/config.json
done

