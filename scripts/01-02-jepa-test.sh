SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh


git_root=$(git rev-parse --show-toplevel)
export PYTHONPATH=$git_root/geojepa
conda activate geojepa

args="$@"
echo "Running with extra args: $args"

ckpt=geojepa_t_jan04
save_name=geojepa_t_jan04_avgmax

append=false

python src/backbone-gen-emb.py model=geojepa_probe \
  model.backbone.ckpt=ckpts/"$ckpt" data=pretraining \
  model.head.in_dim=256 cls_only=true cls='cls' save_name="$save_name" \
  data.batch_size=12 data.group_size=12 append=$append fast_fail=true || exit 1
