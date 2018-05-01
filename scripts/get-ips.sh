nodes=(
ec2-54-145-49-12.compute-1.amazonaws.com
ec2-54-91-21-47.compute-1.amazonaws.com
ec2-54-172-162-139.compute-1.amazonaws.com
ec2-34-201-217-224.compute-1.amazonaws.com
ec2-54-84-91-228.compute-1.amazonaws.com
ec2-52-91-139-75.compute-1.amazonaws.com
ec2-54-152-238-186.compute-1.amazonaws.com
ec2-54-234-106-123.compute-1.amazonaws.com
ec2-54-84-254-69.compute-1.amazonaws.com
ec2-204-236-252-111.compute-1.amazonaws.com
)


for i in "${!nodes[@]}";
do
    echo \"`dig +short ${nodes[$i]}`\",
done

