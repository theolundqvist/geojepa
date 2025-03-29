#!/bin/bash
# Check for and activate CONDA
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh

# Set python path
export PYTHONPATH=$git_root/geojepa

# Reset git repo before running
reset=false
if [[ " $* " == *" --reset "* ]]; then
  reset=true
fi

# Rebuild all cpp projects
build=false
if [[ " $* " == *" --build "* ]]; then
  build=true
fi


#print parameters
echo "reset: $reset"
echo "build: $build"
sleep 0.5
# git variables
git_root=$(git rev-parse --show-toplevel)
git_origin=$(git config --get remote.origin.url)
repo_name=$(basename $git_root)

# make sure all files in the repo are owned by the current user
USER=$(whoami)
GROUP=$(id -gn)

DIR=$git_root
if find "$DIR" ! -user "$USER" -o ! -group "$GROUP" | grep -q .; then
  echo "Some files in $DIR are not owned by $USER:$GROUP. Updating ownership..."
  sudo chown -R "$USER":"$GROUP" "$DIR"
else
  echo "Nice!, All files in $DIR are owned by $USER:$GROUP."
fi




# git reset --hard + git pull + install new requirements
if [ "$reset" = true ]; then
  source $SCRIPT_DIR/reset-git.sh
else
  echo "Cleaning is disabled"
fi


# Compile all C++ projects
if [ "$build" = true ]; then
  bash data/tiles/build_cpp.sh
fi

# Build TINY dataset
bash $SCRIPT_DIR/run-data-pipeline.sh tiny

# Build HUGE dataset
bash $SCRIPT_DIR/run-data-pipeline.sh huge