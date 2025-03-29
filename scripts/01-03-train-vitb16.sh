SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh


git_root=$(git rev-parse --show-toplevel)
export PYTHONPATH=$git_root/geojepa
conda activate geojepa

args="$@"
echo "Running with extra args: $args"

append=false

save_name=vitb16

#python src/backbone-gen-emb.py model=vitb16 \
#  data=pretraining \
#  cls_only=false test_only=false cls='cls' save_name=$save_name \
#  data.batch_size=128 data.group_size=4 data.num_workers=14 append=$append fast_fail=true || exit 1

echo "Embeddings done or failed, sleeping for 5 seconds"
sleep 5

me=$(basename "$0")
logdir=$SCRIPT_DIR/logs
mkdir -p "$logdir"
logfile="$logdir/$me.log"


i=0
cheat_states=("true")
for task in "building_count" "traffic_signals" "max_speed" "car_bridge" "bridge"; do
  for cheat in "${cheat_states[@]}"; do
    echo "Training on task: $task"

    success=false

    python src/train.py model=embedding_lookup \
      model.backbone.dir=data/embeddings/"$save_name"/pretraining_huge \
      model.head.in_dim=768 data=$task data.load_images=false data.cheat=$cheat data.batch_size=256 trainer.accumulate_grad_batches=1 data.group_size=4\
      hparams_search=lr_opt callbacks=no_ckpt +data.persistent_workers=false \
      $args && success=true

    if [ "$success" = false ]; then
      echo "Training failed on task: $task, cheat: $cheat\n" >> $logfile
    else
      echo "Training succeeded on task: $task, cheat: $cheat\n" >> $logfile
    fi
  done
done