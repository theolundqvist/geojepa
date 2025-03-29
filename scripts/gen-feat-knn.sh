SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh


git_root=$(git rev-parse --show-toplevel)
export PYTHONPATH=$git_root/geojepa
conda activate geojepa

args="$@"
echo "Running with extra args: $args"




#names=("geojepa_gti_dec30" "vitb16" "tag_ae_dec_28" "efficientnet" "resnet")





#names=("entity_tag_avg" "geojepa_t_300e_jan13" "geojepa_ti_jan14_300e" "geojepa_gti_170e" "geojepa_t_jan06")

names=("geojepa_t_e1" "geojepa_t_e2" "geojepa_t_e3" "geojepa_gt_300e" "geojepa_gt_300e_small_b")

for model in "${names[@]}"; do
  python pyscripts/h5_knn_feat_html.py -e data/embeddings/"$model"_feat/pretraining_huge -img data/tiles/huge/images -n=200 --tiles=5000 -k=8 --hide-same-tile --save-name=inter_"$model"_feat
done
for model in "${names[@]}"; do
  python pyscripts/h5_knn_feat_html.py -e data/embeddings/"$model"_feat/pretraining_huge -img data/tiles/huge/images -n=1000 --tiles=5000 -k=8
done
