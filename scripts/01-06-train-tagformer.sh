SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh


git_root=$(git rev-parse --show-toplevel)
export PYTHONPATH=$git_root/geojepa
conda activate geojepa

args="$@"
echo "Running with extra args: $args"





#model.head.embedding_file=data/tiles/pruned_embeddings.pkl \

for task in "bridge"; do
  case "$task" in
    building_count)
      min=0
      max="inf"
      ;;
    car_bridge)
      min=0
      max=1
      ;;
    traffic_signals)
      min=0
      max="inf"
      ;;
    max_speed)
      min=-100
      max="inf"
      ;;
    bridge)
      min=0
      max=1
      ;;
  esac

  for cheat in "true" "false"; do
    echo "Training on task: $task with cheat: $cheat (min=$min, max=$max)"

    # ORIGINAL TAGFORMER
    python src/train.py model=tagformer data=$task data.cheat=$cheat \
      model.compile=false \
      model.min_value=$min \
      model.max_value=$max \
      hparams_search=tagformer_lr_opt callbacks=no_ckpt \
      model.head.use_semantic_encoding=true \
      model.head.use_positional_encoding=false \
      $args


    # TAGFORMER WITH POSITIONAL ENCODING
    python src/train.py model=tagformer data=$task data.cheat=$cheat \
      model.compile=false \
      model.min_value=$min \
      model.max_value=$max \
      hparams_search=tagformer_lr_opt  callbacks=no_ckpt \
      model.head.use_semantic_encoding=true \
      model.head.use_positional_encoding=true \
      $args

    # TAGFORMER WITHOUT SEMANTIC EMBEDDING
    python src/train.py model=tagformer data=$task data.cheat=$cheat \
      model.compile=false \
      model.min_value=$min \
      model.max_value=$max \
      hparams_search=tagformer_lr_opt  callbacks=no_ckpt \
      model.head.use_semantic_encoding=false \
      model.head.use_positional_encoding=false \
      $args

    # TAGFORMER WITHOUT SEMANTIC EMBEDDING AND WITH POSITIONAL ENCODING
    python src/train.py model=tagformer data=$task data.cheat=$cheat \
      model.compile=false \
      model.min_value=$min \
      model.max_value=$max \
      hparams_search=tagformer_lr_opt  callbacks=no_ckpt \
      model.head.use_semantic_encoding=false \
      model.head.use_positional_encoding=true \
      $args

    # TAG_COUNT_ENCODER
    python src/train.py model=tagcountencoder data=$task data.cheat=$cheat \
      model.min_value=$min \
      model.max_value=$max \
      hparams_search=lr_opt  callbacks=no_ckpt \
      $args

  done
done
