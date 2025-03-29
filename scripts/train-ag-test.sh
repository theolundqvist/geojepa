# Check if argument is provided
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/ensure-conda.sh

huge=$SCRIPT_DIR/../data/tiles/huge
tiny=$SCRIPT_DIR/../data/tiles/tiny

tasks=$(find "$huge/tasks/" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)
echo "Tasks: $tasks"
time=900

taskgen=$SCRIPT_DIR/../data/task_generator
ag=$SCRIPT_DIR/../data/autogluon

conda activate ag
task="traffic_signals"
echo "\n\n BUILDING $task \n\n"
"$taskgen"/builds/linux_x64/ag_generator -i "$huge"/tasks/"$task" -o $ag/huge/"$task" -c "$taskgen"/pruner_output/pruned_ag20.csv
"$taskgen"/builds/linux_x64/ag_generator -i "$tiny"/tasks/"$task" -o $ag/tiny/"$task" -c "$taskgen"/pruner_output/pruned_ag20.csv

#echo "\n\n TRAINING MEDIUM MODELS on $task \n\n"
#python -m pyscripts.ag_train --task=$task --quality=medium --time-limit=$time --tags --images
#python -m pyscripts.ag_train --task=$task --quality=medium --time-limit=$time --images
#python -m pyscripts.ag_train --task=$task --quality=medium --time-limit=$time --tags
#
#echo "\n\n CLEANING UP \n\n"
#rm -rf $ag/huge/"$task"
#rm -rf $ag/tiny/"$task"
