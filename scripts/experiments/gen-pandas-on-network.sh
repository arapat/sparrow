BASE_DIR="/home/ubuntu"
IDENT_FILE="$BASE_DIR/jalafate-dropbox.pem"

readarray -t nodes < $BASE_DIR/neighbors.txt

echo "Start running pandas?"
read enter

for i in "${!nodes[@]}";
do
    url=${nodes[$i]}

    echo
    echo "===== Running on $url ====="

    echo "Generating data frame on $url"
    ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url \
    "(sudo apt-get install -y python3-pip;
      sudo pip3 install pandas;
      cd $BASE_DIR/rust-boost; git pull; cd $BASE_DIR;
      python3 rust-boost/notebooks/python/gen-pandas.py $i)"
# 2> /dev/null 1>&2 < /dev/null &"

    echo
done

