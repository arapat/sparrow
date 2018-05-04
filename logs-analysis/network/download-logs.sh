DOWNLOAD_PATH="/hdd/nips/may-03/good-10-nodes"
IDENT="~/Dropbox/documents/vault/aws/jalafate-dropbox.pem"
LOG_PATH="~/rust-boost/*.log"
MODEL_PATH="~/rust-boost/model*.json"


readarray -t nodes < ./neighbors.txt

cd $DOWNLOAD_PATH

for i in "${!nodes[@]}";
do
    echo ${nodes[$i]}
    mkdir node-$i
    scp -o StrictHostKeyChecking=no -i $IDENT ubuntu@${nodes[i]}:$LOG_PATH node-$i/ &
    scp -o StrictHostKeyChecking=no -i $IDENT ubuntu@${nodes[i]}:$MODEL_PATH node-$i/ &
done

wait

