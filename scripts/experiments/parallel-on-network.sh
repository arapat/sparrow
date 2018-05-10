ITERATION=$1
FEATURES=564
BASE_DIR="/mnt"
GIT_BRANCH="adaboost"
readarray -t nodes < /home/ubuntu/neighbors.txt

NUM_NODES=${#nodes[@]}
WORK_LOAD=$((($FEATURES+$NUM_NODES-1)/$NUM_NODES))

IDENT_FILE=/home/ubuntu/jalafate-dropbox.pem

if [[ $# -eq 0 ]] ; then
    echo "Please provide the number of iterations."
    exit 1
fi
if [ ! -f $IDENT_FILE ]; then
    echo "Identification file not found!"
    exit 1
fi
 
echo
cat $BASE_DIR/rust-boost/config.json
echo
echo "$NUM_NODES machines. $ITERATION iterations.
Ready to launch?"
read enter


SETUP_COMMAND="
killall rust-boost;
cd $BASE_DIR/rust-boost;
rm *.bin *.log model-*.json"

for i in `seq 1 $NUM_NODES`; do
    url=${nodes[$((i - 1))]}

    echo
    echo "===== Building $url ====="

    scp -o StrictHostKeyChecking=no -i $IDENT_FILE $BASE_DIR/rust-boost/config.json ubuntu@$url:$BASE_DIR/rust-boost/config.json
    ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url "
        $SETUP_COMMAND;
        cd $BASE_DIR/rust-boost && git checkout -- . && git fetch --all &&
        git checkout $GIT_BRANCH && git pull;
        cargo build --release 2> /dev/null 1>&2 < /dev/null &"
    echo
done

ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url "
    $SETUP_COMMAND;
    cd $BASE_DIR/rust-boost && git checkout -- . && git fetch --all &&
    git checkout $GIT_BRANCH && git pull;
    cargo build --release"

for i in `seq 1 $NUM_NODES`; do
    NAME="Node-$i"
    BEGI=$((i * WORK_LOAD - WORK_LOAD))
    FINI=$((i * WORK_LOAD))
    if [ "$i" == "$NUM_NODES" ]; then
        FINI=$FEATURES
    fi

    url=${nodes[$((i - 1))]}

    echo
    echo "===== Launching on $url ====="
    echo "Parameters: $NAME, $BEGI, $FINI, $ITERATION"

    ssh -n -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url "
        cd $BASE_DIR/rust-boost;
        RUST_LOG=DEBUG nohup cargo run --release $NAME $BEGI $FINI $ITERATION 2> run-network.log 1>&2 < /dev/null &"
    echo "Launched."
    echo
done

