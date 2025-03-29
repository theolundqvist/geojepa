SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh


git_root=$(git rev-parse --show-toplevel)
export PYTHONPATH=$git_root/geojepa
conda activate geojepa

args="$@"
echo "Running with extra args: $args"







ckpt=tag_ae_dec_28
append=false

python src/backbone-gen-emb.py model=tag_ae_transfer \
model.backbone.ckpt=ckpts/"$ckpt" data=pretraining \
cls_only=true cls='cls' save_name="$ckpt" \
data.batch_size=32 data.group_size=2 append=$append || exit 1

#for task in "building_count" "car_bridge" "traffic_signals" "max_speed" "bridge"; do
#  for cheat in "true" "false"; do
#    echo "Training on task: $task"
#
#    python src/train.py model=embedding_lookup \
#      model.backbone.dir=data/embeddings/"$ckpt"/pretraining_huge \
#      model.head.in_dim=256 data=$task data.load_images=false data.cheat=$cheat \
#      hparams_search=lr_opt callbacks=no_ckpt +data.persistent_workers=false \
#      $args
#  done
#done
