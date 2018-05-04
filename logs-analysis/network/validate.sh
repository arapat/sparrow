nodes=(
ec2-52-90-142-12.compute-1.amazonaws.com
)

export IDENT_FILE="~/Dropbox/documents/vault/aws/jalafate-dropbox.pem"

cd ../../

for i in "${!nodes[@]}";
do
    url=${nodes[$i]}
    echo "Validating on $url"
    ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url \
    "cd /mnt/rust-boost;
     export output=\$(ls -rt ./model-* | tail -1);
     nohup ./scripts/validate.sh \$output ./validate.log 2> /dev/null 1>&2 < /dev/null &"
#     echo "scripts/validate.sh logs-analysis/network/node-$i/model-9*.json logs-analysis/network/node-$i/validate.log"
#     scripts/validate.sh logs-analysis/network/20-nodes/node-$i/model-9*.json logs-analysis/network/20-nodes/node-$i/validate.log
done

