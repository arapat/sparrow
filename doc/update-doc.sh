#!/bin/bash

if [ ! -f Cargo.toml ]; then
    echo "Please run this script under the project root."
    exit 1
fi

cargo doc --no-deps
git checkout gh-pages
rm -rf implementors/ sparrow/ src/
mv target/doc/* .
git add .
git commit -m "updates"
git push
git checkout master
