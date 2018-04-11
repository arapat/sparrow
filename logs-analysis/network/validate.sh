
cd ../../

for i in {0..4}
do
    echo "scripts/validate.sh logs-analysis/network/node-$i/model-509.json logs-analysis/network/node-$i/validate.log"
    scripts/validate.sh logs-analysis/network/node-$i/model-509.json logs-analysis/network/node-$i/validate.log
done

