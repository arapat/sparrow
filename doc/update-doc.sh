#!/bin/bash

if [ ! -f Cargo.toml ]; then
    echo "Please run this script under the project root."
    exit 1
fi

cargo doc --no-deps
mkdir -p target/doc/images
cp -r doc/graphviz/*.png target/doc/images
echo "Continue?"
read
git checkout gh-pages
rm -rf implementors/ sparrow/ src/
mv target/doc/* .
git add .
git commit -m "updates"
git push
git checkout master
