# Check if argument is provided
if [ -z "$1" ]; then
    echo "Error: No argument provided. Please specify 'tiny' or 'huge'."
    exit 1
fi

# Check if argument is valid
if [ "$1" != "tiny" ] && [ "$1" != "huge" ]; then
    echo "Error: Invalid argument '$1'. Please specify 'tiny' or 'huge'."
    exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh

dataset=$1


git_root=$(git rev-parse --show-toplevel)
cd $git_root/geojepa
# Prepare data pipelines
echo "Preparing data pipelines..."
python data/tiles/gen-makefile.py --all

if [ "$dataset" == "huge" ]; then
  # Find tiles in "remote" storage (local repo/../storage)
  sat_tiles_dir=$git_root/../storage/sat_tiles
  unprocessed_dir=$git_root/../storage/unprocessed
  if [ ! -d "$sat_tiles_dir" ]; then
    echo "Error: $sat_tiles_dir does not exist"
    exit 1
  fi

  if [ ! -d "$unprocessed_dir" ]; then
    echo "Error: $unprocessed_dir does not exist"
    exit 1
  fi
  rm -rf data/tiles/huge/sat_tiles
  cp -r "$sat_tiles_dir" data/tiles/huge/
  cp -r "$unprocessed_dir" data/tiles/huge/
fi

(
  git_root=$(git rev-parse --show-toplevel)
  cd $git_root/geojepa/data/tiles/$dataset
  conda activate geojepa && \
  # 4. process tiles fully (make reset_git && make all)
  make clean
  make all && \
  # 5. generate datasets for all tasks
  make build_h5_all
)

echo "\n\n OSM and SAT pipeline finished \n\n"