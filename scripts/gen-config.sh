BASE_DIR="/mnt"
CONFIG_PATH=$BASE_DIR/rust-boost/config.json

readarray -t nodes < /home/ubuntu/neighbors.txt

echo "{
    \"data_dir\": \"$BASE_DIR\",
    \"network\": [" > $CONFIG_PATH

for i in "${!nodes[@]}";
do
    if [ "$((i + 1))" == "${#nodes[@]}" ]; then
        LE=""
    else
        LE=","
    fi
    echo "        \"`dig +short ${nodes[$i]}`\"$LE" >> $CONFIG_PATH
done

echo "    ]
}" >> $CONFIG_PATH

cat $CONFIG_PATH
