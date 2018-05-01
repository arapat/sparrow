nodes=(
ec2-54-145-49-12.compute-1.amazonaws.com
ec2-54-91-21-47.compute-1.amazonaws.com
ec2-54-172-162-139.compute-1.amazonaws.com
ec2-34-201-217-224.compute-1.amazonaws.com
ec2-54-84-91-228.compute-1.amazonaws.com
ec2-52-91-139-75.compute-1.amazonaws.com
ec2-54-152-238-186.compute-1.amazonaws.com
ec2-54-234-106-123.compute-1.amazonaws.com
ec2-54-84-254-69.compute-1.amazonaws.com
ec2-204-236-252-111.compute-1.amazonaws.com
)

ITERATION=1000
FEATURES=564
NUM_NODES=${#nodes[@]}
WORK_LOAD=$((($FEATURES+$NUM_NODES-1)/$NUM_NODES))

IDENT_FILE=~/jalafate-dropbox.pem

SETUP_COMMAND="killall rust-boost; cd /mnt/rust-boost; rm *.bin *.log model-*.json"

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
    echo
    echo "Parameters: $NAME, $BEGI, $FINI, $ITERATION"

    scp -o StrictHostKeyChecking=no -i $IDENT_FILE /mnt/rust-boost/config.json ubuntu@$url:/mnt/rust-boost/config.json
    ssh -n -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url "
        $SETUP_COMMAND;
        RUST_LOG=DEBUG nohup cargo run --release $NAME $BEGI $FINI $ITERATION 2> run-network.log 1>&2 < /dev/null &"
done
echo
cat /mnt/rust-boost/config.json
echo

