SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh


git_root=$(git rev-parse --show-toplevel)
export PYTHONPATH=$git_root/geojepa
conda activate geojepa

args="$@"

#ckpts=("geojepa_t_jan04" "geojepa_gti_dec30" "vitb16" "tag_ae_dec_28")
#ckpts=("tag_ae_dec_28")
#ckpts=("geojepa_t_jan04_avgmax" "tag_ae_dec_28")

#num_jobs=4
#
#mkdir -p logs/sklearn

#ckpts=("geojepa_ti_jan06" "geojepa_t_jan04" "geojepa_gti_dec30" "vitb16" "tag_ae_dec_28" "efficientnet" "resnet" "scalemae")
ckpts=("geojepa_t_e1" "geojepa_t_e2" "geojepa_t_e3" "geojepa_gt_300e" "geojepa_gt_300e_small_b")
#ckpts=("geojepa_gti_dec30" "vitb16" "tag_ae_dec_28" "efficientnet" "resnet" "scalemae")
for i in {0..5}; do
  for ckpt in "${ckpts[@]}"; do
    python pyscripts/torch_transfer.py --ckpt $ckpt
  done
done
#
#{
#for ckpt in "${ckpts[@]}"; do
#  for task in "${tasks[@]}"; do
#      echo "python pyscripts/torch_transfer.py --ckpt $ckpt"
#  done
#done
#} | xargs -P $num_jobs -I {} sh -c '{}' 2>&1 | tee logs/sklearn/transfer_all.log


