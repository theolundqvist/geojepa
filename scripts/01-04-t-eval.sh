SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh


git_root=$(git rev-parse --show-toplevel)
export PYTHONPATH=$git_root/geojepa
conda activate geojepa

args="$@"
echo "Running with extra args: $args"

limit=10000
append=false

ckpt=geojepa_t_jan04
save_name=test_t_jan04
#python src/backbone-gen-emb.py model=geojepa_probe \
#  model.backbone.ckpt=ckpts/"$ckpt" data=pretraining \
#  model.head.in_dim=256 cls_only=true test_only=true cls='cls' limit=$limit save_name="$save_name" \
#  data.batch_size=6 data.group_size=12 append=$append fast_fail=true || exit 1
#python pyscripts/h5_knn_html.py -e data/embeddings/$save_name/pretraining_huge -img data/tiles/huge/images --name $ckpt -n=200
#
#
#python src/backbone-gen-emb.py model=geojepa_probe \
#  model.backbone.ckpt=ckpts/"$ckpt" data=pretraining \
#  model.head.in_dim=256 cls_only=true test_only=false cls='cls' save_name="$ckpt"\
#  data.batch_size=6 data.group_size=12 append=$append fast_fail=true || exit 1
#
#echo "Embeddings done or failed, sleeping for 5 seconds"

#cheat_states=("true")
#for task in "building_count" "traffic_signals" "max_speed" "car_bridge" "bridge"; do
#  for cheat in "${cheat_states[@]}"; do
#    echo "Training on task: $task"
#
#    python src/train.py model=embedding_lookup \
#      model.backbone.dir=data/embeddings/"$ckpt"/pretraining_huge \
#      model.head.in_dim=512 data=$task data.load_images=false data.cheat=$cheat data.batch_size=256 trainer.accumulate_grad_batches=1 data.group_size=4\
#      hparams_search=lr_opt callbacks=no_ckpt +data.persistent_workers=false \
#      $args
#  done
#done
ckpt=geojepa_t_jan04
save_name=test_ckpt_load
limit=10
python src/backbone-gen-emb.py model=geojepa_probe \
  model.backbone.ckpt=ckpts/"$ckpt" data=pretraining \
  model.head.in_dim=256 cls_only=true test_only=true cls='cls' limit=$limit save_name="$save_name" \
  data.batch_size=6 data.group_size=12 append=$append fast_fail=true || exit 1
