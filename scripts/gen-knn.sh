SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh


git_root=$(git rev-parse --show-toplevel)
export PYTHONPATH=$git_root/geojepa
conda activate geojepa

args="$@"
echo "Running with extra args: $args"


#names=("tagformer_lmae_100e" "geojepa_gti_300e" "entity_tag_avg" "geojepa_t_300e_jan13" "geojepa_ti_jan14_300e")
names=("geojepa_t_e1" "geojepa_t_e2" "geojepa_t_e3" "geojepa_gt_300e" "geojepa_gt_300e_small_b")
for model in "${names[@]}"; do
  python pyscripts/h5_knn_html.py -e data/embeddings/$model/pretraining_huge -img data/tiles/huge/images --name "$model" -n=300 --limit=25000 -k=8
done
exit 0

names=("vitb16" "tag_ae_dec_28" "scalemae" "tagformer_lmae" "tagformer_ae")

for model in "${names[@]}"; do
  #python pyscripts/h5_knn_html.py -e data/embeddings/$model/pretraining_huge -img data/tiles/huge/images --name selected_"$model" --limit=25000 -k=8 --only="16_18845_24995,16_11431_26084"
  python pyscripts/h5_knn_html.py -e data/embeddings/$model/pretraining_huge -img data/tiles/huge/images --name $model -n=300 --limit=25000 -k=8
done

