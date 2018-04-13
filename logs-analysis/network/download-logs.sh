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

IDENT="~/Dropbox/documents/vault/aws/jalafate-dropbox.pem"
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

