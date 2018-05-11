# RUST_BACKTRACE=1 RUST_LOG=DEBUG cargo run validate $1 2> $2
RUST_LOG=DEBUG cargo run --release validate $1 $2 2> $3

