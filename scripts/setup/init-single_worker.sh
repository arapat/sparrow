export GIT_REPO="https://github.com/arapat/rust-boost.git"
export GIT_BRANCH="aws-scale"

sudo umount /mnt
# Choose one of the two options according to the instance type:
# 1. One SSD
export DISK="/dev/xvdb"
yes | sudo mkfs.ext4 -L MY_DISK $DISK
# 2. Two SSDs - RAID
# yes | sudo mdadm --create --verbose /dev/md0 --level=0 --name=MY_RAID --raid-devices=2 /dev/xvdb /dev/xvdc
# yes | sudo mkfs.ext4 -L MY_DISK /dev/md0

sudo apt-get update
sudo apt-get install -y awscli

sudo mount LABEL=MY_DISK /mnt
sudo chown -R ubuntu /mnt
cd /mnt

git config --global user.name "Julaiti Alafate"
git config --global user.email "jalafate@gmail.com"
git config --global push.default simple

echo "export EDITOR=vim" >> ~/.bashrc

git clone $GIT_REPO /mnt/rust-boost

yes | sudo apt-get install cargo

wait 5

mkdir ~/.aws
cp ~/credentials ~/.aws/
aws s3 cp s3://ucsd-data/splice/testing.bin .
aws s3 cp s3://ucsd-data/splice/training.bin .
