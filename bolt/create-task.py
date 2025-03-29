from prompt_toolkit import prompt

name = prompt("What should the task be called: ", placeholder="test_sat_img_encoder")

if not name.strip():
    print("Name is required")
    exit(1)


module = prompt("Enter the Python module to run: ", default="src.")

import shutil
from pathlib import Path

base_dir = Path(__file__).parent.resolve()

template_path = base_dir / "tasks/template"
new_task_path = base_dir / f"tasks/{name}"

shutil.copytree(template_path, new_task_path)

config_file = new_task_path / "config.yaml"
with config_file.open("r+") as f:
    content = f.read()
    f.seek(0)
    f.write(f"name: '{name}'\n" + content)

run_file = new_task_path / "run.sh"
with run_file.open("a") as f:
    f.write(f"\npython -m {module}\n")

print("Task created at", new_task_path)
print("To submit the task, execute the following command:\n")
print(f"bash bolt/tasks/{name}/submit.sh")
