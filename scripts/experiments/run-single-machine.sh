killall rust-boost
cd /mnt/rust-boost/
rm *.bin *.log model-*.json

echo "===== Configuration ====="
echo
cat config.json
echo

for i in `seq 8 35`; do
    head=$((280 - i * 8))
    tail=$((284 + i * 8))
    echo "Testing on range [$head, $tail]"
    RUST_LOG=DEBUG cargo run --release single $head $tail 500 2> run-$i.log
    mkdir run-large-$i
    mv model-*.json *.log run-large-$i
    rm *.bin
done
echo

for i in `seq 1 28`; do
    head=$((282 - i * 2))
    tail=$((282 + i * 2))
    echo "Testing on range [$head, $tail]"
    RUST_LOG=DEBUG cargo run --release single $head $tail 500 2> run-$i.log
    mkdir run-large-$i
    mv model-*.json *.log run-large-$i
    rm *.bin
done
echo

