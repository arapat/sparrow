nodes=(
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
