# RUST_BACKTRACE=1 RUST_LOG=DEBUG cargo run validate $1 2> $2
RUST_LOG=DEBUG cargo run --release count > $1 2> /dev/null

