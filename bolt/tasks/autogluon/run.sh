conda activate iris

echo "Testing if autogluon libraries exist, should give data error..."
python -m pyscripts.autogluon_task

dir="pyscripts/"
online="s3://tlundqvist/geojepa/data/autogluon_data.zip"
echo "downloading autogluon data..."
time aws-cli s3 cp --only-show-errors "$online" "$dir"/autogluon_data.zip  || exit 1
time unzip -q "$dir"/autogluon_data.zip || exit 1

python -m pyscripts.autogluon_task

zip -r $CLOUD_ARTIFACT_DIR/artifacts.zip AutogluonModels
echo "Artifacts zipped and available at $CLOUD_ARTIFACT_DIR/artifacts.zip"
