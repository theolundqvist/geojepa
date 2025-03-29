SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh


git_root=$(git rev-parse --show-toplevel)
export PYTHONPATH=$git_root/geojepa
conda activate geojepa

args="$@"
echo "Running with extra args: $args"


ckpts=("tagformer_ae" "tagformer_lmae")
#for ckpt in "${ckpts[@]}"; do
#  # geojepa_probe is actually generic probe, so just set backbone.ckpt to whatever
#  echo "Running for $ckpt"
#  python src/backbone-gen-emb.py model=geojepa_probe \
#    model.backbone.ckpt=ckpts/"$ckpt" data=pretraining model.head.in_dim=768 \
#    cls_only=true test_only=false cls='cls' save_name="$ckpt" \
#    data.batch_size=6 data.group_size=12 append=false fast_fail=true || exit 1
#done

for i in {0..4}; do
  for ckpt in "${ckpts[@]}"; do
    python pyscripts/torch_transfer.py --ckpt $ckpt
  done
done