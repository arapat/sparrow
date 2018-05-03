export TIMEOUT=0.5
BASE_DIR="/home/ubuntu"

killall rust-boost
cd $BASE_DIR/rust-boost/
rm *.bin *.log model-*.json

echo "===== Configuration ====="
echo
cat config.json
echo

echo "Start running?"
read enter

for j in `seq 1 10`; do
for i in `seq 1 282`; do
    head=$((282 - i * 1))
    tail=$((282 + i * 1))
    echo "Testing on range [$head, $tail]"
    RUST_LOG=DEBUG cargo run --release single $head $tail 100 2> run-$i.log
    mkdir timeout-$i
    mv model-*.json timeout-$i/
    mv run-$i.log timeout-$i/run-$i-$j.log
    rm *.log *.bin
done
done
echo

