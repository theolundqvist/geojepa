conda activate iris

dataset="pretraining"

#make sure CLOUD_ARTIFACT_DIR is set
if [ -z "$CLOUD_ARTIFACT_DIR" ]; then
    echo "CLOUD_ARTIFACT_DIR is not set"
    exit 1
fi
# Define common parameters, NOT DATA
common=(
    experiment="01-07-geojepa-gt"
    paths=bolt
    callbacks=bolt
    +trainer.enable_progress_bar=false
    logger=tensorboard
    trainer.log_every_n_steps=48
    paths.log_dir="$CLOUD_ARTIFACT_DIR"/logs
    # file lookup is suuuuuuper slow on bolt
    model.tokenizer.img_encoder._target_=src.modules.vision_backbones.ViTB16
    model.tokenizer.img_encoder_selector._target_=src.modules.vision_backbones.ImageSelector
    #data.load_images=true
)

# Define hyperparameters
hparams=()

# if file does not existss
if [ ! -f "data/tiles/huge/images.h5" ]; then

  # download pretrained geometry model
  aws-cli s3 cp --only-show-errors s3://tlundqvist/geojepa/polygnn-ckpt-dec-26 src/models/pretrained/ || exit 1

  dataset_dir="data/tiles/tiny"
  dir="data/tiles/tiny/tasks"
  archive="$dataset".tar.zst
  online="s3://tlundqvist/geojepa/data/tiny"
  pbf_archive="$online""/tasks/"$archive
  image_archive="$online"/images.h5.tar.zst
  echo "downloading tiny..."
  mkdir -p "$dir"
  time aws-cli s3 cp --only-show-errors  "$pbf_archive"   "$dir"/"$archive"  || exit 1
  time tar --use-compress-program="zstd -d -T0" -xf "$dir"/"$archive" -C "data/tiles/tiny/" || exit 1
  time aws-cli s3 cp --only-show-errors  "$image_archive"  "$dataset_dir"/images.h5.tar.zst  || exit 1
  time tar --use-compress-program="zstd -d -T0" -xf "$dataset_dir"/images.h5.tar.zst -C "data/tiles/tiny/" || exit 1
  echo "tree -L 1 $dir"
  echo "tree -L 1 $dataset_dir"
  tree -L 2 "$dir"
  tree -L 3 "$dataset_dir"


  # run fast dev run
  python src/train.py \
      "${common[@]}" \
      data="$dataset" \
      data.size=tiny \
      data.group_size=2 \
      "${hparams[@]}" \
      callbacks.model_checkpoint.save_last=false \
      trainer.max_epochs=1 \
      model.compile=false \
      trainer.accumulate_grad_batches=1 \
      data.batch_size=6 || exit 1


  echo "Success running tiny."

  dataset_dir="data/tiles/huge"
  dir="data/tiles/huge/tasks"
  archive="$dataset".tar.zst
  online="s3://tlundqvist/geojepa/data/huge"
  pbf_archive="$online""/tasks/"$archive
  image_archive="$online"/images.h5
  echo "downloading huge..."
  mkdir -p "$dir"
  set -e

  # Run each command, capturing and printing errors
  {
      time aws-cli s3 cp --only-show-errors "$pbf_archive" "$dir/$archive"
  } || {
      echo "Error in aws-cli s3 cp for $pbf_archive" >&2
      exit 1
  }

  {
      time tar --use-compress-program="zstd -d -T0" -xf "$dir/$archive" -C "data/tiles/huge/"
  } || {
      echo "Error extracting archive $dir/$archive" >&2
      exit 1
  }

  {
      time aws-cli s3 cp --only-show-errors "$image_archive" "$dataset_dir/images.h5"
  } || {
      echo "Error in aws-cli s3 cp for $image_archive" >&2
      exit 1
  }

  echo "tree -L 1 $dir"
  echo "tree -L 1 $dataset_dir"
  tree -L 2 "$dir"
  tree -L 3 "$dataset_dir"

fi


# Use find with -printf to list modification times and paths, sort, and extract the newest file
ckpt_path=$(
  find "$CLOUD_ARTIFACT_DIR/logs" -type f -name "last.ckpt" -printf "%T@ %p\n" \
  | sort -n \
  | tail -1 \
  | cut -d' ' -f2-
)
# Check if a checkpoint was found
if [ -z "$ckpt_path" ]; then
  echo "No last.ckpt file found."
  ckpt_path=null
else
  echo "Continuing from latest checkpoint: $ckpt_path"
fi

python src/train.py \
  "${common[@]}" \
  data="$dataset" \
  "${hparams[@]}" \
  ckpt_path="$ckpt_path"


  #model.tokenizer.img_encoder.dir=data/embeddings/vitb16/pretraining_huge \

zip -r $CLOUD_ARTIFACT_DIR/artifacts.zip $CLOUD_ARTIFACT_DIR/logs
echo "Artifacts zipped and available at $CLOUD_ARTIFACT_DIR/artifacts.zip"
