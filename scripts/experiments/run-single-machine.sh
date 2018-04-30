export TIMEOUT=5

killall rust-boost
cd /mnt/rust-boost/
rm *.bin *.log model-*.json

echo "===== Configuration ====="
echo
cat config.json
echo

for j in `seq 1 10`; do
for i in `seq 8 35`; do
    head=$((280 - i * 8))
    tail=$((284 + i * 8))
    echo "Testing on range [$head, $tail]"
    RUST_LOG=DEBUG timeout $TIMEOUT cargo run --release single $head $tail 100 2> run-$i.log
    mkdir timeout-large-$i
    mv model-*.json timeout-large-$i/
    mv run-$i.log timeout-large-$i/run-$i-$j.log
    rm *.log *.bin
done
done
echo

for j in `seq 1 10`; do
for i in `seq 1 28`; do
    head=$((282 - i * 2))
    tail=$((282 + i * 2))
    echo "Testing on range [$head, $tail]"
    RUST_LOG=DEBUG timeout $TIMEOUT cargo run --release single $head $tail 100 2> run-$i.log
    mkdir timeout-small-$i
    mv model-*.json timeout-small-$i/
    mv run-$i.log timeout-small-$i/run-$i-$j.log
    rm *.log *.bin
done
done
echo

