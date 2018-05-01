nodes=(
ec2-54-172-121-2.compute-1.amazonaws.com
ec2-54-237-240-178.compute-1.amazonaws.com
ec2-54-236-141-208.compute-1.amazonaws.com
ec2-34-235-135-227.compute-1.amazonaws.com
ec2-54-173-36-174.compute-1.amazonaws.com
ec2-54-85-123-49.compute-1.amazonaws.com
ec2-34-230-8-187.compute-1.amazonaws.com
ec2-184-73-60-48.compute-1.amazonaws.com
ec2-35-173-138-175.compute-1.amazonaws.com
ec2-54-158-129-205.compute-1.amazonaws.com
)

export INIT_SCRIPT="/mnt/rust-boost/scripts/aws/init-c3_xlarge-ubuntu.sh"
export IDENT_FILE="~/jalafate-dropbox.pem"
export GIT_REPO="https://github.com/arapat/rust-boost.git"
export GIT_BRANCH="aws"

for i in "${!nodes[@]}";
do
    url=${nodes[$i]}
    echo "Checking $url..."

    if ssh -o StrictHostKeyChecking=no -i $IDENT_FILE $url test -f /mnt/training.bin \> /dev/null 2\>\&1
    then
        :
    else
        echo "!!! Training does not exists on $url."
    fi
done
