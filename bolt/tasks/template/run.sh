conda activate iris

dataset="{{DATASET_NAME}}"

# Define common parameters, NOT DATA
common=(
    trainer=gpu
    experiment={{EXPERIMENT_NAME}}
    paths=bolt
    callbacks=bolt
)

# Define hyperparameters
hparams=(
  {{OVERRIDES}}
)


dir="data/tiles/tiny/tasks"
archive="$dataset".tar.zst
online="s3://tlundqvist/geojepa/data/tiny/tasks"
echo "downloading tiny..."
mkdir -p "$dir"
time aws-cli s3 cp --only-show-errors  "$online"/"$archive"   "$dir"/"$archive"  || exit 1
time tar --use-compress-program="zstd -d -T0" -xf "$dir"/"$archive" -C "data/tiles/tiny/" || exit 1
echo "tree $dir"
tree "$dir"


# run fast dev run
python src/train.py \
    "${common[@]}" \
    "${hparams[@]}" \
    data="$dataset" \
    data.tiny=true \
    debug=fdr \
    trainer=gpu || exit 1

dir="data/tiles/huge/tasks"
archive="$dataset".tar.zst
online="s3://tlundqvist/geojepa/data/huge/tasks"
echo "downloading huge..."
mkdir -p "$dir"
time aws-cli s3 cp --only-show-errors  "$online"/"$archive"   "$dir"/"$archive"  || exit 1
time tar --use-compress-program="zstd -d -T0" -xf "$dir"/"$archive" -C "data/tiles/huge/" || exit 1
echo "tree $dir"
tree "$dir"

# train
python src/train.py \
  data="$dataset" \
  "${common[@]}" \
  "${hparams[@]}"

zip -r $CLOUD_ARTIFACT_DIR/artifacts.zip logs
echo "Artifacts zipped and available at $CLOUD_ARTIFACT_DIR/artifacts.zip"
