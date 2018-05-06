BASE_DIR="/home/ubuntu"
IDENT_FILE="$BASE_DIR/jalafate-dropbox.pem"

readarray -t nodes < $BASE_DIR/neighbors.txt

echo "Start validating?"
read enter

for i in "${!nodes[@]}";
do
    url=${nodes[$i]}
    echo "Validating on $url"
    ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url \
    "cd $BASE_DIR/rust-boost;
     export output=\$(ls -rt ./model-* | tail -1);
     nohup ./scripts/validate.sh \$output ./validate.log 2> /dev/null 1>&2 < /dev/null &"
done

