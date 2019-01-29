#!/bin/bash
git checkout gh-pages
rm -rf implementors/ sparrow/ src/
mv target/doc/* .
git add .
git commit -m "updates"
git push
git checkout master
