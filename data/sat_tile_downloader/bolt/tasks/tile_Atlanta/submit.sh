# copy the current task to .task to simplify the submission and remote code
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${script_dir%/*/*/*}"
mkdir -p "$root"/.task
cp -r "$script_dir"/* "$root"/.task/
cp "$root/builds/linux_x64/merge_tiles" "$root/.task/"
cp "$root/download.py" "$root/.task/"
cp "$root/tile.sh" "$root/.task/"
{TASK_TOOL} submit --config "$root"/.task/config.yaml --tar "$root/.task"
rm -r "$root"/.task
