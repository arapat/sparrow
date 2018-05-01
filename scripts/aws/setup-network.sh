nodes=(
ec2-34-230-68-167.compute-1.amazonaws.com
ec2-54-173-173-8.compute-1.amazonaws.com
ec2-34-238-168-230.compute-1.amazonaws.com
ec2-54-145-49-12.compute-1.amazonaws.com
ec2-34-235-163-2.compute-1.amazonaws.com
ec2-54-86-52-193.compute-1.amazonaws.com
ec2-54-91-21-47.compute-1.amazonaws.com
ec2-54-172-162-139.compute-1.amazonaws.com
ec2-18-232-62-108.compute-1.amazonaws.com
ec2-52-90-71-139.compute-1.amazonaws.com
ec2-34-201-217-224.compute-1.amazonaws.com
ec2-54-84-91-228.compute-1.amazonaws.com
ec2-52-91-139-75.compute-1.amazonaws.com
ec2-54-173-162-25.compute-1.amazonaws.com
ec2-54-152-238-186.compute-1.amazonaws.com
ec2-54-89-44-26.compute-1.amazonaws.com
ec2-54-234-106-123.compute-1.amazonaws.com
ec2-54-84-254-69.compute-1.amazonaws.com
ec2-54-89-134-139.compute-1.amazonaws.com
ec2-204-236-252-111.compute-1.amazonaws.com
)

export INIT_SCRIPT="/mnt/rust-boost/scripts/aws/init-two_ssd.sh"
export IDENT_FILE="~/jalafate-dropbox.pem"
export GIT_REPO="https://github.com/arapat/rust-boost.git"
export GIT_BRANCH="aws-scale"

if [ $1 = "init" ]; then
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
                scp -o StrictHostKeyChecking=no -i $IDENT_FILE /mnt/training.bin ubuntu@$url:/mnt &
            fi
            if ssh -o StrictHostKeyChecking=no -i $IDENT_FILE $url test -f /mnt/testing.bin \> /dev/null 2\>\&1
            then
                echo "Testing file exists."
            else
                scp -o StrictHostKeyChecking=no -i $IDENT_FILE /mnt/testing.bin ubuntu@$url:/mnt &
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

    echo
    echo "Now waiting for training/testing files to be transmitted to all other computers..."
    echo
    wait
fi

# Build package
if [ $1 = "build" ]; then
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

    echo "Package build was executed."
fi
