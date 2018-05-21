BASE_DIR="/mnt"
IDENT_FILE="/home/ubuntu/jalafate-dropbox.pem"
RESET="reset"

readarray -t nodes < /home/ubuntu/neighbors.txt

echo "Start validating?"
read enter

for i in "${!nodes[@]}";
do
    url=${nodes[$i]}
    echo
    echo "Validating on $url"
    ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url \
    "cd $BASE_DIR/rust-boost;
     killall rust-boost;
     rm ./validate.log;
     git pull;
     nohup ./scripts/validate.sh $RESET ./ ./validate.log 2> /dev/null 1>&2 < /dev/null &"
done

