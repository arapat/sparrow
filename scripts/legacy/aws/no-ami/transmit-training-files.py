import os
import subprocess
import sys
import time

log_path = "/home/ubuntu/transmit.log"
base_dir = "/mnt"
neighbors = "neighbors.txt"
ident_file = "/home/ubuntu/jalafate-dropbox.pem"
# NOTE: we assume all computers have this file at this path
this_file = "/mnt/rust-boost/scripts/aws/transmit-training-files.py"
is_master = (sys.argv[1].strip() == "true")

f_log = open(log_path, 'w')


def write_log(s):
    f_log.write(s + '\n')
    f_log.flush()


childs = []

neighbor_path = os.path.join(base_dir, neighbors)
with open(neighbor_path) as f:
    remain = f.readlines()
remain = [line.strip() for line in remain]
remain = [line for line in remain if line]
write_log("Read %d neighbors." % len(remain))

cur_remote = None
next_remote = None
proc_send_file = None

if is_master:
    write_log("This node is the master node. Initiating the transferring process.")
    time.sleep(1)
    for addr in remain:
        command = "ssh -o StrictHostKeyChecking=no -i {} {} rm {}" \
                        .format(ident_file, addr, neighbor_path)
        write_log("\nRunning `{}`\n".format(command))
        proc_init = subprocess.Popen(command.split())
        proc_init.wait()
    with open(neighbor_path, 'w') as f:
        f.write('\n'.join(remain))

while childs or remain or proc_send_file is not None:
    if proc_send_file is None and remain:
        next_remote = remain[0]
        remain = remain[1:]
        write_log("Processing 1 neighbor. Remaining neighbors: %d" % len(remain))
        command = "ssh -o StrictHostKeyChecking=no -i {} {} test -f {}" \
                        .format(ident_file, next_remote, neighbor_path)
        write_log("\nRunning `{}`\n".format(command))
        proc_check = subprocess.Popen(command.split())
        proc_check.wait()
        if proc_check.returncode == 0:
            write_log("{} has already been working.".format(next_remote))
            continue
        command = "ssh -o StrictHostKeyChecking=no -i {} {} test -f /mnt/testing.bin" \
                        .format(ident_file, next_remote)
        write_log("\nRunning `{}`\n".format(command))
        proc_check_file = subprocess.Popen(command.split())
        proc_check_file.wait()
        if proc_check_file.returncode == 0:
            cur_remote = next_remote
            write_log("{} has already obtained files.".format(next_remote))
        else:
            command = "scp -o StrictHostKeyChecking=no -i {} /mnt/training.bin /mnt/testing.bin ubuntu@{}:/mnt" \
                            .format(ident_file, next_remote)
            write_log("\nRunning `{}`\n".format(command))
            proc_send_file = subprocess.Popen(command.split())
            write_log("Sending files to {}...".format(next_remote))

    if proc_send_file is not None and proc_send_file.poll() is not None:
        write_log("Files have been sent to {} with returncode {}"
                      .format(next_remote, proc_send_file.returncode))
        cur_remote = next_remote
        proc_send_file = None

    if cur_remote is not None and len(remain) >= 2:
        write_log("launching new worker at {}.".format(cur_remote))
        # 1. Send neighbor file
        mid = int(len(remain) / 2)
        filepath = "/tmp/load_{}.txt".format(cur_remote)
        with open(filepath, 'w') as f:
            f.write('\n'.join(remain[:mid]))
        command = "scp -o StrictHostKeyChecking=no -i {} {} ubuntu@{}:{}" \
                        .format(ident_file, filepath, cur_remote, neighbor_path)
        write_log("\nRunning `{}`\n".format(command))
        proc_send_neighbor = subprocess.Popen(command.split())
        proc_send_neighbor.wait()
        # 2. Send pem file
        command = "ssh -o StrictHostKeyChecking=no -i {} {} test -f {}" \
                        .format(ident_file, next_remote, ident_file)
        write_log("\nRunning `{}`\n".format(command))
        proc_check_ident = subprocess.Popen(command.split())
        proc_check_ident.wait()
        if proc_check_ident.returncode != 0:
            command = "scp -o StrictHostKeyChecking=no -i {} {} ubuntu@{}:{}" \
                            .format(ident_file, ident_file, cur_remote, ident_file)
            write_log("\nRunning `{}`\n".format(command))
            proc_send_neighbor2 = subprocess.Popen(command.split())
            proc_send_neighbor2.wait()
        # 3. Launch worker
        command = "ssh -o StrictHostKeyChecking=no -i {} {} python3 {} false" \
                        .format(ident_file, cur_remote, this_file)
        write_log("\nRunning `{}`\n".format(command))
        proc_helper = subprocess.Popen(command.split())
        childs.append((cur_remote, proc_helper))
        write_log("Launched a new helper at {} with {} neighbors.".format(cur_remote, mid))

        cur_remote = None
        remain = remain[mid:]
        write_log("Splitted neighbors with workers. Remaining neighbors: %d" % len(remain))

    new_childs = []
    for url, p in childs:
        if p.poll() is not None:
            write_log("{} is exited with the returncode {}.".format(url, p.returncode))
        else:
            new_childs.append((url, p))
    childs = new_childs

    time.sleep(2)

f_log.close()
