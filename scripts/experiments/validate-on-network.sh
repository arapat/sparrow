BASE_DIR="/home/ubuntu"
IDENT_FILE="~/Dropbox/documents/vault/aws/jalafate-dropbox.pem"

readarray -t nodes < $BASE_DIR/neighbors.txt

for i in "${!nodes[@]}";
do
    url=${nodes[$i]}
    echo "Validating on $url"
    ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url \
    "cd $BASE_DIR/rust-boost;
     export output=\$(ls -rt ./model-* | tail -1);
     nohup ./scripts/validate.sh \$output ./validate.log 2> /dev/null 1>&2 < /dev/null &"
done

