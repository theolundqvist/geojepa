SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

git_root=$(git rev-parse --show-toplevel)
cd $git_root/geojepa

args="$@"
echo "Running with extra args: $args"

ckpts=(
  "geojepa_gti_170e"
  "geojepa_gti_300e"
  "geojepa_gt_300e_small_b"
  "geojepa_gt_300e"
  "geojepa_ti_jan14_300e"
  "geojepa_t_e1"
  "geojepa_t_e2"
  "geojepa_t_e3"
  "geojepa_t_jan06"
  "tagformer_lmae_100e"
  "entity_tag_avg"
)
for ckpt in "${ckpts[@]}"; do
  echo "Downloading embeddings for $ckpt"
  mkdir data/embeddings/"$ckpt"
  mkdir data/embeddings/"$ckpt"_feat
  ./download.sh data/embeddings/"$ckpt" --progress
  ./download.sh data/embeddings/"$ckpt"_feat --progress
done

