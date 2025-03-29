SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh


git_root=$(git rev-parse --show-toplevel)
export PYTHONPATH=$git_root/geojepa
conda activate geojepa

args="$@"
echo "Running with extra args: $args"




#names=("geojepa_ti_jan06" "geojepa_t_jan04" "geojepa_gti_dec30" "vitb16" "tag_ae_dec_28" "efficientnet" "resnet" "scalemae")
#names=("geojepa_gti_dec30" "vitb16" "tag_ae_dec_28" "efficientnet" "resnet")

model="entity_tag_avg"

#python src/backbone-gen-emb.py model=$model \
#  data=pretraining  \
#  cls_only=true test_only=false cls='cls' save_name="$model" \
#  data.batch_size=32 data.group_size=12 append=false fast_fail=true || exit 1
#
#for i in {0..4}; do
#    python pyscripts/torch_transfer.py --ckpt "$model"
#done

tasks=("max_speed" "building_count" "bridge" "car_bridge" "traffic_signals")
for task in "${tasks[@]}"; do
#  python src/backbone-gen-emb.py model=$model \
#    data=$task data.cheat=false  \
#    cls_only=true test_only=false cls='cls' save_name="$model"_masked_"$task" \
#    data.batch_size=32 data.group_size=10 append=false fast_fail=true || exit 1
  for i in {0..4}; do
      python pyscripts/torch_transfer.py --ckpt "$model"_masked_"$task" --task $task
  done
  #rm -rf data/embeddings/"$model"_masked/"$task"_huge
done

#python pyscripts/h5_knn_html.py -e data/embeddings/$model/pretraining_huge -img data/tiles/huge/images --name $model -n=200 --limit=25000
#
#python src/backbone-gen-emb.py model=$model \
#  data=pretraining  \
#  cls_only=false test_only=true cls='cls' limit=10000 save_name="$model"_feat \
#  data.batch_size=6 data.group_size=12 append=false fast_fail=true || exit 1
#
#python pyscripts/h5_knn_feat_html.py -e data/embeddings/"$model"_feat/pretraining_huge -img data/tiles/huge/images -n=400 --limit=100000
#
#python src/backbone-gen-emb.py model=$model \
#  data=pretraining \
#  cls_only=false test_only=true cls='cls' limit=1000 save_name="$model"_feat_1k \
#  data.batch_size=6 data.group_size=12 append=false fast_fail=true || exit 1
#
#python pyscripts/h5_knn_feat_html.py -e data/embeddings/"$model"_feat_1k/pretraining_huge -img data/tiles/huge/images -n=200 --limit=100000
