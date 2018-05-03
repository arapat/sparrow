ITERATION=1000
FEATURES=564
BASE_DIR="/home/ubuntu"
readarray -t nodes < $BASE_DIR/neighbors.txt

NUM_NODES=${#nodes[@]}
WORK_LOAD=$((($FEATURES+$NUM_NODES-1)/$NUM_NODES))

IDENT_FILE=$BASE_DIR/jalafate-dropbox.pem

if [ ! -f $IDENT_FILE ]; then
    echo "Identification file not found!"
    exit 1
fi
 
echo
cat $BASE_DIR/rust-boost/config.json
echo
echo "Ready to launch on $NUM_NODES machines?"
read enter


SETUP_COMMAND="killall rust-boost; cd $BASE_DIR/rust-boost; rm *.bin *.log model-*.json"

for i in `seq 1 $NUM_NODES`; do
    url=${nodes[$((i - 1))]}

    echo
    echo "===== Building $url ====="

    scp -o StrictHostKeyChecking=no -i $IDENT_FILE $BASE_DIR/rust-boost/config.json ubuntu@$url:$BASE_DIR/rust-boost/config.json
    ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url "
        $SETUP_COMMAND;
        cd $BASE_DIR/rust-boost && git checkout -- . && git fetch --all &&
        git checkout $GIT_BRANCH && git pull;
        cargo build --release"
    echo
done


for i in `seq 1 $NUM_NODES`; do
    NAME="Node-$i"
    BEGI=$((i * WORK_LOAD - WORK_LOAD))
    FINI=$((i * WORK_LOAD))
    if [ "$i" == "$NUM_NODES" ]; then
        FINI=$FEATURES
    fi

    echo
    echo "===== Launching on $url ====="
    echo "Parameters: $NAME, $BEGI, $FINI, $ITERATION"
    echo

    ssh -n -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url "
        RUST_LOG=DEBUG nohup cargo run --release $NAME $BEGI $FINI $ITERATION 2> run-network.log 1>&2 < /dev/null &"
done

