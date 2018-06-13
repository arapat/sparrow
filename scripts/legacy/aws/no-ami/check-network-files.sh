nodes=(
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
