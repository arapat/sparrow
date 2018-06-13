BASE_DIR="/home/ubuntu"
export GIT_REPO="https://github.com/arapat/rust-boost.git"
export GIT_BRANCH="aws-scale"
export IDENT_FILE="$BASE_DIR/jalafate-dropbox.pem"

readarray -t nodes < $BASE_DIR/neighbors.txt

# Build package
for i in "${!nodes[@]}";
do
url=${nodes[$i]}
echo "Building $url"

ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url "
    cd $BASE_DIR/rust-boost && git checkout -- . && git fetch --all &&
    git checkout $GIT_BRANCH && git pull &&
    cargo build --release" && \
scp -o StrictHostKeyChecking=no -i $IDENT_FILE $BASE_DIR/rust-boost/config.json ubuntu@$url:$BASE_DIR/rust-boost/config.json &
done

wait

echo "Package build was executed."

