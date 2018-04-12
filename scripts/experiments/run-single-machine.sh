killall rust-boost
cd /mnt/rust-boost/
rm *.bin *.log model-*.json

for i in `seq 1 35`; do
    head=$((280 - i * 8))
    tail=$((284 + i * 8))
    echo "Testing on range [$head, $tail]"
    RUST_LOG=DEBUG cargo run --release single $head $tail 500 2> run-$i.log
    mkdir run$i
    mv *.json model-*.log run$i
done

