SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh


git_root=$(git rev-parse --show-toplevel)
export PYTHONPATH=$git_root/geojepa
conda activate geojepa

args="$@"
echo "Running with extra args: $args"

# ORIGINAL TAGFORMER
#python src/train.py experiment=train_tagformer_ae \
#  model.compile=false \
#  trainer.max_epochs=30 \
#  data.batch_size=14 \
#  $args

python src/train.py experiment=train_tagformer_lmae \
  model.compile=false \
  trainer.max_epochs=100 \
  data.batch_size=14 \
  $args
