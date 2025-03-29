SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh


git_root=$(git rev-parse --show-toplevel)
export PYTHONPATH=$git_root/geojepa
conda activate geojepa

args="$@"
echo "Running with extra args: $args"

ckpts=("geojepa_gti_170e" "geojepa_gti_300e")
tasks=("max_speed" "building_count" "bridge" "car_bridge" "traffic_signals")
for ckpt in "${ckpts[@]}"; do
  for task in "${tasks[@]}"; do
    python src/backbone-gen-emb.py model=geojepa_probe \
      model.backbone.ckpt=ckpts/"$ckpt" data=$task model.head.in_dim=768 \
      cls_only=true test_only=false cls='cls' data.load_images=true save_name="$ckpt"_masked_"$task" \
      data.batch_size=6 data.group_size=12 append=false fast_fail=true
    #rm -rf data/embeddings/"$model"_masked/"$task"_huge
  done
done

for ckpt in "${ckpts[@]}"; do
  for task in "${tasks[@]}"; do
    for i in {0..4}; do
        python pyscripts/torch_transfer.py --ckpt "$ckpt"_masked_"$task" --task $task
    done
  done
done
exit 0

