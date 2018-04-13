sudo umount /mnt
yes | sudo mdadm --create --verbose /dev/md0 --level=0 --name=MY_RAID --raid-devices=2 /dev/xvdca /dev/xvdcb
sudo mkfs.ext4 -L MY_RAID /dev/md0
sudo mount LABEL=MY_RAID /mnt
sudo chown -R ubuntu /mnt

git config --global user.name "Julaiti Alafate"
git config --global user.email "jalafate@gmail.com"

echo "export EDITOR=vim" >> ~/.bashrc

