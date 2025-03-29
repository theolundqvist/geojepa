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
task_path = base_dir / f"tasks/{name}"

shutil.copytree(template_path, task_path)

config_file = task_path / "config.yaml"
with config_file.open("r+") as f:
    content = f.read()
    f.seek(0)
    f.write(f"name: '{name}'\n" + content)

run_file = task_path / "run.sh"
with run_file.open("a") as f:
    f.write(f"\npython -m {module}\n")


iris_reqs = base_dir / "requirements-iris.txt"
task_reqs = base_dir / task_path
import pkg_resources

current_reqs = list(
    str(ws).split()[0] + "==" + str(ws).split()[1] for ws in pkg_resources.working_set
)
current_reqs_names = list(str(ws).split()[0] for ws in pkg_resources.working_set)
# read iris requirements
with iris_reqs.open("r") as f:
    iris_reqs = f.read().splitlines()

iris_reqs_names = list(req.split("==")[0] for req in iris_reqs)

needed_reqs = []
for i, req_name in enumerate(current_reqs_names):
    if req_name not in iris_reqs_names:
        needed_reqs.append(current_reqs[i])
    else:
        for j, iris_req_name in enumerate(iris_reqs_names):
            if req_name == iris_req_name:
                needed_reqs.append(iris_reqs[j])
                break

print("Current env requirements:", len(current_reqs))
print("Docker packages:", len(iris_reqs))

with (task_reqs / "requirements.txt").open("w") as f:
    for req in needed_reqs:
        f.write(req + "\n")

print(
    "Created requirements.txt with",
    len(needed_reqs),
    "initial packages needed to replicate the current environment",
)

print("Task created at", task_path)
print("To submit the task, execute the following command:\n")
print(f"bash bolt/tasks/{name}/submit.sh")
