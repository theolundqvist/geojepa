# Check if argument is provided
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh


logs=$SCRIPT_DIR/../logs

find $logs -type f -name "results.json" -exec git add -f "{}" \;
find $logs -type f -name "config.yaml" ! -wholename "*hydra/config.yaml" -exec git add -f "{}" \;
find $SCRIPT_DIR/../pyscripts/logs/ag -type f -name "results.json" -exec git add -f "{}" \;
