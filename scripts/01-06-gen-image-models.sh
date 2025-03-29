SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh


git_root=$(git rev-parse --show-toplevel)
export PYTHONPATH=$git_root/geojepa
conda activate geojepa

args="$@"
echo "Running with extra args: $args"



ckpts=("tagformer_lmae_100e")
for ckpt in "${ckpts[@]}"; do
  # geojepa_probe is actually generic probe, so just set backbone.ckpt to whatever
  echo "Running for $ckpt"
  python src/backbone-gen-emb.py model=geojepa_probe \
    model.backbone.ckpt=ckpts/"$ckpt" data=pretraining model.head.in_dim=256 \
    cls_only=true test_only=false cls='cls' save_name="$ckpt" \
    data.batch_size=6 data.group_size=12 append=false fast_fail=true || exit 1

  #python pyscripts/h5_knn_html.py -e data/embeddings/$ckpt/pretraining_huge -img data/tiles/huge/images --name $ckpt -n=200 --limit=25000

  python src/backbone-gen-emb.py model=geojepa_probe  \
    model.backbone.ckpt=ckpts/"$ckpt" data=pretraining model.head.in_dim=256 \
    cls_only=false test_only=true cls='cls' limit=5000 save_name="$ckpt"_feat \
    data.batch_size=6 data.group_size=12 append=false fast_fail=true || exit 1

  #python pyscripts/h5_knn_feat_html.py -e data/embeddings/"$ckpt"_feat/pretraining_huge -img data/tiles/huge/images -n=200 --limit=100000
done

for i in {0..4}; do
  for ckpt in "${ckpts[@]}"; do
    python pyscripts/torch_transfer.py --ckpt $ckpt
  done
done

exit 0


ckpts=("geojepa_t_300e_jan13" "geojepa_ti_jan14_300e" "geojepa_gti_170e")
tasks=("max_speed" "building_count" "bridge" "car_bridge" "traffic_signals")
for ckpt in "${ckpts[@]}"; do
  for task in "${tasks[@]}"; do
    python src/backbone-gen-emb.py model=geojepa_probe \
      model.backbone.ckpt=ckpts/"$ckpt" data=$task model.head.in_dim=768 \
      cls_only=true test_only=false cls='cls' save_name="$ckpt"_masked_"$task" \
      data.batch_size=6 data.group_size=12 append=false fast_fail=true || exit 1
    #rm -rf data/embeddings/"$model"_masked/"$task"_huge
  done
done

for ckpt in "${ckpts[@]}"; do
  for task in "${tasks[@]}"; do
    for i in {0..4}; do
        python pyscripts/torch_transfer.py --ckpt "$model"_masked_"$task" --task $task
    done
  done
done
exit 0

