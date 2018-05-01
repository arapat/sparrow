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

export INIT_SCRIPT="/mnt/rust-boost/scripts/aws/init-two_ssd.sh"
export IDENT_FILE="~/jalafate-dropbox.pem"
export GIT_REPO="https://github.com/arapat/rust-boost.git"
export GIT_BRANCH="aws-scale"

if [[ $# -eq 0 ]] ; then
    echo "Initialize all computers. Are you sure? (y/N)"
    read yesno
    if [[ "$yesno" != "y" ]] ; then
        echo "Aborted."
        exit 1
    fi

    for i in "${!nodes[@]}";
    do
        url=${nodes[$i]}
        echo
        echo "===== Initializing $url ====="
        echo

        if ssh -o StrictHostKeyChecking=no -i $IDENT_FILE $url test -f /mnt/init-done.txt \> /dev/null 2\>\&1
        then
            echo "The node has been initialized. Skipped."
        else
            # Copy init script
            scp -o StrictHostKeyChecking=no -i $IDENT_FILE $INIT_SCRIPT ubuntu@$url:~/init.sh

            # Execute init script
            ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url sudo bash ~/init.sh

            # Copy data
            if ssh -o StrictHostKeyChecking=no -i $IDENT_FILE $url test -f /mnt/training.bin \> /dev/null 2\>\&1
            then
                echo "Training file exists."
            else
                scp -o StrictHostKeyChecking=no -i $IDENT_FILE /mnt/training.bin ubuntu@$url:/mnt
            fi
            if ssh -o StrictHostKeyChecking=no -i $IDENT_FILE $url test -f /mnt/testing.bin \> /dev/null 2\>\&1
            then
                echo "Testing file exists."
            else
                scp -o StrictHostKeyChecking=no -i $IDENT_FILE /mnt/testing.bin ubuntu@$url:/mnt
            fi

            # Clone repository
            ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url git clone $GIT_REPO /mnt/rust-boost

            # Install cargo
            ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url sudo apt-get update
            ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url sudo apt-get install -y cargo

            ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url touch /mnt/init-done.txt
            echo "Initialization is completed."
        fi
    done
fi

# Build package
for i in "${!nodes[@]}";
do
    url=${nodes[$i]}
    echo "Building $url"

    ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url "
        cd /mnt/rust-boost && git checkout -- . && git fetch --all &&
        git checkout $GIT_BRANCH && git pull &&
        cargo build --release" && \
    scp -o StrictHostKeyChecking=no -i $IDENT_FILE /mnt/rust-boost/config.json ubuntu@$url:/mnt/rust-boost/config.json &
done

wait

init=" NOT"
if [[ $# -eq 0 ]] ; then
    init=""
fi
echo "Initialization was$init executed."
echo "Package build was executed."

