echo "Cleaning is enabled"
set -e
git fetch origin main
# remove all local changes
git reset --hard
# remove all untracked files
git clean -fd
# pull latest changes
git pull

# install any new requirements
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $script_dir/ensure-conda.sh
conda activate geojepa

# installs using torch 2.4.0, which is not compatible with intel macs
pip install -r requirements.txt
pip install torch_geometric
pip install pyg_lib torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
