SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh


git_root=$(git rev-parse --show-toplevel)
export PYTHONPATH=$git_root/geojepa
conda activate geojepa

args="$@"
echo "Running with extra args: $args"

limit=10000
append=false

ckpt=vitb16
save_name=test_vitb16
#python src/backbone-gen-emb.py model=vitb16 \
#  data=pretraining \
#  cls_only=true test_only=true cls='cls' limit=$limit save_name="$save_name" \
#  data.batch_size=6 data.group_size=12 append=$append fast_fail=true || exit 1
python pyscripts/h5_knn_html.py -e data/embeddings/$save_name/pretraining_huge -img data/tiles/huge/images --name $ckpt -n=200

ckpt=tag_ae_jan_02
save_name=test_tag_ae
#python src/backbone-gen-emb.py model=tag_ae_transfer \
#  model.backbone.ckpt=ckpts/"$ckpt" data=pretraining \
#  cls_only=true test_only=true cls='cls' limit=$limit save_name="$save_name" \
#  data.batch_size=6 data.group_size=12 append=$append fast_fail=true || exit 1
python pyscripts/h5_knn_html.py -e data/embeddings/$save_name/pretraining_huge -img data/tiles/huge/images --name $ckpt -n=200

ckpt=geojepa_t_dec30
save_name=test_t
#python src/backbone-gen-emb.py model=geojepa_probe \
#  model.backbone.ckpt=ckpts/"$ckpt" data=pretraining \
#  model.head.in_dim=256 cls_only=true test_only=true cls='cls' limit=$limit save_name="$save_name" \
#  data.batch_size=6 data.group_size=12 append=$append fast_fail=true || exit 1
python pyscripts/h5_knn_html.py -e data/embeddings/$save_name/pretraining_huge -img data/tiles/huge/images --name $ckpt -n=200

ckpt=geojepa_t_jan04
save_name=test_t_jan04
#python src/backbone-gen-emb.py model=geojepa_probe \
#  model.backbone.ckpt=ckpts/"$ckpt" data=pretraining \
#  model.head.in_dim=256 cls_only=true test_only=true cls='cls' limit=$limit save_name="$save_name" \
#  data.batch_size=6 data.group_size=12 append=$append fast_fail=true || exit 1
python pyscripts/h5_knn_html.py -e data/embeddings/$save_name/pretraining_huge -img data/tiles/huge/images --name $ckpt -n=200


ckpt=geojepa_gti_dec30
save_name=test_gti
#python src/backbone-gen-emb.py model=geojepa_probe \
#  model.backbone.ckpt=ckpts/"$ckpt" data=pretraining \
#  model.head.in_dim=256 cls_only=true test_only=true cls='cls' limit=$limit save_name="$save_name" \
#  data.batch_size=6 data.group_size=12 append=$append fast_fail=true || exit 1
python pyscripts/h5_knn_html.py -e data/embeddings/$save_name/pretraining_huge -img data/tiles/huge/images --name $ckpt -n=200


