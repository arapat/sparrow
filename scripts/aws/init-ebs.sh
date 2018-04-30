export GIT_BRANCH=master

git config --global user.name "Julaiti Alafate"
git config --global user.email "jalafate@gmail.com"

echo "export EDITOR=vim" >> ~/.bashrc

# For solo
sudo apt-get update
sudo apt-get install -y cargo

cd /mnt/rust-boost
git checkout -- .
git fetch --all
git checkout $GIT_BRANCH
git pull
cargo build --release

