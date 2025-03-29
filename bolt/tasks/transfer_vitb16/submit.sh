
# =============================================================================
# Automatically generated. Do not modify directly.
# Use the corresponding Python script (submit_experiment.py) to regenerate if necessary.
# =============================================================================
# copy the current task to .task to simplify the submission and remote code
task_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${task_dir%/*/*/*}"
mkdir -p "$root"/.task
cp -r "$task_dir"/* "$root"/.task/

# what to include in the task
cp -r "$root/.project-root" "$root/.task/"
cp -r "$root/configs" "$root/.task/"
cp -r "$root/src" "$root/.task/"
rm -r "$root/.task/src/models/pretrained"
mkdir -p "$root/.task/data/tiles"
cp "$root/data/tiles/embeddings.pkl" "$root/.task/data/tiles/"
# ----------------------

echo "---------   files   ---------"
tree "$root"/.task
echo "---------   config   ---------"
cat "$root"/.task/config.yaml
echo "-------  entry point   ------"
cat "$root"/.task/run.sh
echo "-----------------------------"
echo "Checking lightning config..."

if [ -z "$1" ]; then
  CLOUD_ARTIFACT_DIR="$root/.task/pre-submit-artifacts" \
  python "$root"/.task/src/check_config.py \
  trainer=gpu \
  experiment=transfer_vitb16 \
  data=traffic_signals_tiny \
  paths=bolt \
  callbacks=bolt \
  debug=fdr
fi


read -p "Submit this task? [Y]/n " answer
answer=$(echo "$answer" | tr 'A-Z' 'a-z')

# Check the user's response
if [[ -z "$answer" || "$answer" == "y" || "$answer" == "yes" ]]; then
    {TASK_TOOL} submit --config "$root"/.task/config.yaml --tar "$root/.task"
else
    echo "Exiting..."
fi

rm -r "$root"/.task
