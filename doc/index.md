# Sparrow

Sparrow is an impementation of boosting that is optimized for training on very large datasets, as well as training in limited memory settings. The opimizations involve two technologies: early stopping and selective sampling. Please read [our paper](https://arxiv.org/abs/1901.09047) for more details.

## Prerequisites

1. Install Rust following the instruction on [the Rust offical webiste](https://www.rust-lang.org/tools/install).
2. Compile Sparrow: `cargo build --release`

The Sparrow binary file would be generated at `target/release/sparrow`.

## Usage

Sparrow is written as a Rust library. It also supports running as a binary. The sparrow binary reads configuration from a specified configuration file. Many examples of the configuration files can be found in the [examples/](examples) directory.

To run the Sparrow binary, please provide the path to the configuration file.

For training,
```bash
./sparrow train <path to the config file>
```

For testing (or prediction),
```bash
./sparrow test <path to the config file>
```

To use Sparrow as a Rust library, please read [the document here](sparrow).
