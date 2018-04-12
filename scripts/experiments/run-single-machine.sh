RUST_LOG=DEBUG cargo run --release 0   564 300  2> run-0.log
mkdir run0
mv *.json *.log run0
RUST_LOG=DEBUG cargo run --release 50  514 500  2> run-1.log
mkdir run1
mv *.json *.log run1
RUST_LOG=DEBUG cargo run --release 100 464 700  2> run-2.log
mkdir run2
mv *.json *.log run2
RUST_LOG=DEBUG cargo run --release 150 414 1000 2> run-3.log
mkdir run3
mv *.json *.log run3
RUST_LOG=DEBUG cargo run --release 200 364 1000 2> run-4.log
mkdir run4
mv *.json *.log run4
RUST_LOG=DEBUG cargo run --release 250 314 1000 2> run-5.log
mkdir run5
mv *.json *.log run5
RUST_LOG=DEBUG cargo run --release 260 304 1000 2> run-6.log
mkdir run6
mv *.json *.log run6
RUST_LOG=DEBUG cargo run --release 270 294 1000 2> run-7.log
mkdir run7
mv *.json *.log run7

