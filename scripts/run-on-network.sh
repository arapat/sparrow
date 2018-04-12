nodes=(
ec2-54-172-121-2.compute-1.amazonaws.com
ec2-54-237-240-178.compute-1.amazonaws.com
ec2-54-236-141-208.compute-1.amazonaws.com
ec2-34-235-135-227.compute-1.amazonaws.com
ec2-54-173-36-174.compute-1.amazonaws.com
ec2-54-85-123-49.compute-1.amazonaws.com
ec2-34-230-8-187.compute-1.amazonaws.com
ec2-184-73-60-48.compute-1.amazonaws.com
ec2-35-173-138-175.compute-1.amazonaws.com
ec2-54-158-129-205.compute-1.amazonaws.com
)
ITERATION=500
FEATURES=564
NUM_NODES=${#nodes[@]}
WORK_LOAD=$((($FEATURES+$NUM_NODES-1)/$NUM_NODES))

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

    ssh -n -o StrictHostKeyChecking=no -i /mnt/jalafate-dropbox.pem ubuntu@$url "
        $SETUP_COMMAND;
        RUST_LOG=DEBUG nohup cargo run --release $NAME $BEGI $FINI $ITERATION 2> run-network.log 1>&2 < /dev/null &"
done
echo

