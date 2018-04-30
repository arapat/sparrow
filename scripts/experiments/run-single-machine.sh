export TIMEOUT=5

killall rust-boost
cd /mnt/rust-boost/
rm *.bin *.log model-*.json

echo "===== Configuration ====="
echo
cat config.json
echo

# for j in `seq 1 10`; do
# for i in `seq 8 35`; do
#     head=$((280 - i * 8))
#     tail=$((284 + i * 8))
#     echo "Testing on range [$head, $tail]"
#     RUST_LOG=DEBUG timeout $TIMEOUT cargo run --release single $head $tail 100 2> run-$i.log
#     mkdir timeout-large-$i
#     mv model-*.json timeout-large-$i/
#     mv run-$i.log timeout-large-$i/run-$i-$j.log
#     rm *.log *.bin
# done
# done
# echo

for j in `seq 1 10`; do
for i in `seq 1 282`; do
    head=$((282 - i * 1))
    tail=$((282 + i * 1))
    echo "Testing on range [$head, $tail]"
    RUST_LOG=DEBUG timeout $TIMEOUT cargo run --release single $head $tail 100 2> run-$i.log
    mkdir timeout-$i
    mv model-*.json timeout-$i/
    mv run-$i.log timeout-$i/run-$i-$j.log
    rm *.log *.bin
done
done
echo

