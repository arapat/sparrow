export USER="ubuntu"
export SERVER="54.172.121.2"

pwd
scp -r -i ~/Dropbox/documents/vault/aws/jalafate-dropbox.pem $USER@$SERVER:/mnt/rust-boost/run* .
cd ../../

for i in {1..35}
do
    echo "../../scripts/validate.sh run$i/model-300.json run$i/validate.log"
    scripts/validate.sh logs-analysis/local/run$i/model-300.json logs-analysis/local/run$i/validate.log
done

