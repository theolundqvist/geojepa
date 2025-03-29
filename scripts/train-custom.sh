SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh

git_root=$(git rev-parse --show-toplevel)
export PYTHONPATH=$git_root/geojepa

conda activate geojepa

# Train tag model with access to all osm data
python src/train.py experiment=transfer_vision_models
python src/train.py experiment=train_tag_transformer
