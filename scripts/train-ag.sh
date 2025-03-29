# Check if argument is provided
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh

huge=$SCRIPT_DIR/../data/tiles/huge
small=$SCRIPT_DIR/../data/tiles/small

time=3600

taskgen=$SCRIPT_DIR/../data/task_generator
ag=$SCRIPT_DIR/../data/autogluon

conda activate ag
#for task in "building_count" "car_bridge" "traffic_signals" "max_speed" "bridge"; do
for task in "car_bridge"; do
  echo "\n\n BUILDING $task \n\n"
  "$taskgen"/builds/linux_x64/ag_generator -i "$huge"/tasks/"$task" -o $ag/huge/"$task" -c "$taskgen"/pruner_output/pruned_ag20.csv
  "$taskgen"/builds/linux_x64/ag_generator -i "$small"/tasks/"$task" -o $ag/small/"$task" -c "$taskgen"/pruner_output/pruned_ag20.csv
  echo "\n\n TRAINING MEDIUM MODELS on $task \n\n"
  python -m pyscripts.ag_train --task=$task --quality=medium --time-limit=600 --images
  python -m pyscripts.ag_train --task=$task --quality=medium --time-limit=900 --images

  echo "\n\n TRAINING HIGH QUALITY MODELS on $task \n\n"
  python -m pyscripts.ag_train --task=$task --quality=high --time-limit=600 --images
  python -m pyscripts.ag_train --task=$task --quality=high --time-limit=900 --images

  echo "\n\n TRAINING BEST QUALITY MODELS on $task \n\n"
  python -m pyscripts.ag_train --task=$task --quality=best --time-limit=3600 --images

  echo "\n\n CLEANING UP \n\n"
  rm -rf $ag/huge/"$task"
  rm -rf $ag/small/"$task"
done
