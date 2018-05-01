import os
import subprocess
import time

base_dir = "/mnt"
neighbors = "neighbors.txt"
ident_file = "~/jalafate-dropbox.pem"
# NOTE: we assume all computers have this file at this path
this_file = "/mnt/rust-boost/scripts/aws/transmit-training-files.py"

childs = []

neighbor_path = os.path.join(base_dir, neighbors)
with open(neighbor_path) as f:
    remain = f.readlines()
remain = [line.strip() for line in remain]
remain = [line for line in remain if line]

cur_remote = None
next_remote = None
proc_send_file = None

while childs or remain:
    if proc_send_file is None and remain:
        next_remote = remain[0]
        remain = remain[1:]
        command = "ssh -o StrictHostKeyChecking=no -i {} {} test -f /mnt/training.bin" \
                        .format(ident_file, next_remote)
        proc_check_file = subprocess.Popen(command.split())
        proc_check_file.wait()
        if proc_check_file.returncode == 0:
            cur_remote = next_remote
            print("{} has already obtained files.".format(next_remote))
        else:
            command = "scp -o StrictHostKeyChecking=no -i {} /mnt/*.bin ubuntu@{}:/mnt" \
                            .format(ident_file, next_remote)
            proc_send_file = subprocess.Popen(command.split())
            print("Sending files to {}...".format(next_remote))

    if proc_send_file is not None and proc_send_file.poll() is None:
        print("Files have been sent to {}.".format(next_remote))
        cur_remote = next_remote
        proc_send_file = None

    if cur_remote is not None and len(remain) >= 2:
        mid = int(len(remain) / 2)
        with open("/tmp/load.txt", 'w') as f:
            f.write('\n'.join(remain[:mid]))
        command = "scp -o StrictHostKeyChecking=no -i {} /tmp/load.txt ubuntu@{}:{}" \
                        .format(ident_file, cur_remote, neighbor_path)
        proc_send_neighbor = subprocess.Popen(command.split())
        proc_send_neighbor.wait()
        command = "python3 {}".format(this_file)
        proc_helper = subprocess.Popen(command.split())
        childs.append((cur_remote, proc_helper))
        print("Launched a new helper at {}.".format(cur_remote))

        cur_remote = None
        remain = remain[mid:]

    new_childs = []
    for url, p in childs:
        if p.poll() is None:
            print("{} is exited with the returncode {}.".format(url, p.returncode))
        else:
            new_childs.append((url, p))
    childs = new_childs

    time.sleep(2)
