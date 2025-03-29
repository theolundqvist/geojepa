import os
import shutil
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pathlib import Path

from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import confirm
from prompt_toolkit.shortcuts import radiolist_dialog


experiment_name = prompt("enter experiment name: ", default="transfer_efficientnet")

if not experiment_name.strip():
    print("Experiment is required")
    exit(1)

dataset_name = prompt("enter dataset name: ", default="traffic_signals")

if not dataset_name.strip():
    print("Dataset is required")
    exit(1)

overrides = prompt("Enter overrides: ", default="model.compile=true")
options = [("A100", "1xA100"), ("V100", "1xV100")]
GPU = radiolist_dialog(
    title="GPU Selection",
    text="Select a GPU to use for training:",
    values=options,
).run()

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

template_task_path = Path("bolt/tasks/template")
new_task_path = Path(f"bolt/tasks/{experiment_name}")

if os.path.exists(new_task_path):
    if confirm("Task already exists, overwrite?"):
        shutil.rmtree(new_task_path)
    else:
        exit(0)

shutil.copytree(template_task_path, new_task_path)


def replace_in_file(file, replacements):
    with file.open("r") as f:
        content = f.read()
    for placeholder, actual in replacements.items():
        content = content.replace(placeholder, actual)
    with file.open("w") as f:
        f.write("""
# =============================================================================
# Automatically generated. Do not modify directly.
# Use the corresponding Python script (submit_experiment.py) to regenerate if necessary.
# =============================================================================\n""")
        f.write(content)


replacements = {
    "{{DATASET_NAME}}": dataset_name,
    "{{EXPERIMENT_NAME}}": experiment_name,
    "{{TASK_NAME}}": experiment_name + "_" + GPU,
    "{{OVERRIDES}}": overrides,
}
shutil.move(new_task_path / f"config-{GPU}.yaml", new_task_path / "config.yaml")
replace_in_file(new_task_path / "config.yaml", replacements)
replace_in_file(new_task_path / "run.sh", replacements)
replace_in_file(new_task_path / "submit.sh", replacements)
replace_in_file(new_task_path / "setup.sh", replacements)
replace_in_file(new_task_path / "requirements.txt", replacements)

input("Task has been generated, press Enter to start submission process...")
os.system(f"bash bolt/tasks/{experiment_name}/submit.sh")
