nodes=(
)

IDENT="~/jalafate-dropbox.pem"
LOG_PATH="/mnt/rust-boost/run-network.log"
LOG_PATH="/mnt/rust-boost/*.log"
MODEL_PATH="/mnt/rust-boost/model*.json"

for i in "${!nodes[@]}";
do
    echo ${nodes[$i]}
    mkdir node-$i
    scp -o StrictHostKeyChecking=no -i $IDENT ubuntu@${nodes[i]}:$LOG_PATH node-$i/
    scp -o StrictHostKeyChecking=no -i $IDENT ubuntu@${nodes[i]}:$MODEL_PATH node-$i/
done

