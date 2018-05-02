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
ec2-54-173-173-8.compute-1.amazonaws.com
ec2-34-235-163-2.compute-1.amazonaws.com
ec2-18-232-62-108.compute-1.amazonaws.com
ec2-52-90-71-139.compute-1.amazonaws.com
ec2-54-89-44-26.compute-1.amazonaws.com
ec2-34-230-68-167.compute-1.amazonaws.com
ec2-34-238-168-230.compute-1.amazonaws.com
ec2-54-86-52-193.compute-1.amazonaws.com
ec2-54-173-162-25.compute-1.amazonaws.com
ec2-54-89-134-139.compute-1.amazonaws.com
ec2-52-23-180-58.compute-1.amazonaws.com
ec2-54-175-161-189.compute-1.amazonaws.com
ec2-52-87-189-74.compute-1.amazonaws.com
ec2-34-203-195-64.compute-1.amazonaws.com
ec2-54-173-171-137.compute-1.amazonaws.com
ec2-54-174-40-143.compute-1.amazonaws.com
ec2-54-90-227-212.compute-1.amazonaws.com
ec2-54-152-183-97.compute-1.amazonaws.com
ec2-54-147-192-136.compute-1.amazonaws.com
ec2-54-91-103-96.compute-1.amazonaws.com
ec2-54-90-180-128.compute-1.amazonaws.com
ec2-54-204-201-52.compute-1.amazonaws.com
ec2-52-90-216-234.compute-1.amazonaws.com
ec2-107-23-219-195.compute-1.amazonaws.com
ec2-54-84-117-103.compute-1.amazonaws.com
ec2-54-208-30-58.compute-1.amazonaws.com
ec2-35-170-50-186.compute-1.amazonaws.com
ec2-54-146-42-45.compute-1.amazonaws.com
ec2-35-153-158-177.compute-1.amazonaws.com
ec2-54-91-66-106.compute-1.amazonaws.com
ec2-54-88-17-17.compute-1.amazonaws.com
ec2-34-204-72-21.compute-1.amazonaws.com
ec2-34-229-112-152.compute-1.amazonaws.com
ec2-54-80-175-232.compute-1.amazonaws.com
ec2-54-208-193-171.compute-1.amazonaws.com
ec2-54-236-247-207.compute-1.amazonaws.com
ec2-54-88-152-254.compute-1.amazonaws.com
ec2-54-160-179-3.compute-1.amazonaws.com
ec2-35-153-205-215.compute-1.amazonaws.com
ec2-54-173-174-36.compute-1.amazonaws.com
ec2-54-87-138-124.compute-1.amazonaws.com
ec2-34-226-203-45.compute-1.amazonaws.com
ec2-52-87-199-119.compute-1.amazonaws.com
ec2-52-207-229-103.compute-1.amazonaws.com
ec2-54-242-190-42.compute-1.amazonaws.com
ec2-35-173-223-47.compute-1.amazonaws.com
ec2-54-173-229-43.compute-1.amazonaws.com
ec2-34-235-123-90.compute-1.amazonaws.com
ec2-35-153-162-74.compute-1.amazonaws.com
ec2-52-90-49-86.compute-1.amazonaws.com
ec2-35-153-203-88.compute-1.amazonaws.com
ec2-34-207-240-225.compute-1.amazonaws.com
ec2-54-88-99-169.compute-1.amazonaws.com
ec2-52-87-198-0.compute-1.amazonaws.com
ec2-34-227-67-17.compute-1.amazonaws.com
ec2-54-196-220-119.compute-1.amazonaws.com
ec2-54-175-73-188.compute-1.amazonaws.com
ec2-52-91-251-28.compute-1.amazonaws.com
ec2-54-197-6-159.compute-1.amazonaws.com
ec2-52-55-251-97.compute-1.amazonaws.com
ec2-34-229-102-78.compute-1.amazonaws.com
ec2-52-23-180-7.compute-1.amazonaws.com
ec2-54-237-217-182.compute-1.amazonaws.com
ec2-54-90-163-247.compute-1.amazonaws.com
ec2-34-201-125-109.compute-1.amazonaws.com
ec2-54-209-31-98.compute-1.amazonaws.com
ec2-52-91-142-51.compute-1.amazonaws.com
ec2-34-230-16-237.compute-1.amazonaws.com
ec2-52-201-252-69.compute-1.amazonaws.com
ec2-54-91-82-61.compute-1.amazonaws.com
ec2-34-229-245-159.compute-1.amazonaws.com
ec2-52-207-223-144.compute-1.amazonaws.com
ec2-35-168-10-78.compute-1.amazonaws.com
ec2-54-83-169-63.compute-1.amazonaws.com
ec2-107-23-242-70.compute-1.amazonaws.com
ec2-54-164-236-78.compute-1.amazonaws.com
ec2-54-173-25-194.compute-1.amazonaws.com
ec2-18-232-130-53.compute-1.amazonaws.com
ec2-54-157-31-211.compute-1.amazonaws.com
ec2-54-196-93-155.compute-1.amazonaws.com
ec2-52-204-40-49.compute-1.amazonaws.com
ec2-54-164-20-246.compute-1.amazonaws.com
ec2-54-226-34-153.compute-1.amazonaws.com
ec2-34-228-228-161.compute-1.amazonaws.com
ec2-35-173-204-19.compute-1.amazonaws.com
ec2-34-239-142-15.compute-1.amazonaws.com
ec2-54-83-174-69.compute-1.amazonaws.com
ec2-35-168-11-245.compute-1.amazonaws.com
ec2-54-82-155-27.compute-1.amazonaws.com
ec2-54-174-13-95.compute-1.amazonaws.com
)

export IDENT_FILE="~/jalafate-dropbox.pem"

cd ../../

for i in "${!nodes[@]}";
do
    url=${nodes[$i]}
    echo "Validating on $url"
    ssh -o StrictHostKeyChecking=no -i $IDENT_FILE ubuntu@$url \
    "cd /mnt/rust-boost;
     export output=\$(ls -rt ./model-* | tail -1);
     nohup ./scripts/validate.sh \$output ./validate.log 2> /dev/null 1>&2 < /dev/null &"
done

wait

