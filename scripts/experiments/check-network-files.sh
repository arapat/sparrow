BASE_DIR="/home/ubuntu"
readarray -t nodes < $BASE_DIR/neighbors.txt

export INIT_SCRIPT="/mnt/rust-boost/scripts/aws/init-c3_xlarge-ubuntu.sh"
export IDENT_FILE="~/jalafate-dropbox.pem"
export GIT_REPO="https://github.com/arapat/rust-boost.git"
export GIT_BRANCH="aws"

for i in "${!nodes[@]}";
do
    url=${nodes[$i]}
    echo "Checking $url"

    ssh -o StrictHostKeyChecking=no -i $IDENT_FILE $url "
        ls -rt $BASE_DIR/rust-boost/model* | tail -2;
        ls -rt $BASE_DIR/rust-boost/*.log;
        ls -rt $BASE_DIR/rust-boost/*.pkl;
        ps aux | grep [t]arget/release/rust-boost
    "
    read enter
done
