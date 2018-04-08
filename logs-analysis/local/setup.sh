export USER="ubuntu"
export SERVER=""

pwd
scp -r -i ~/Dropbox/documents/vault/aws/jalafate-dropbox.pem $USER@$SERVER:/mnt/rust-boost/run* .
cd ../../

for i in {0..7}
do
    echo "../../scripts/validate.sh run$i/model-300.json run$i/validate.log"
    scripts/validate.sh logs-analysis/local/run$i/model-300.json logs-analysis/local/run$i/validate.log
done

