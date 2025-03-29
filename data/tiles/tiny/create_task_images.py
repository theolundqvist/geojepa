import os
import shutil
import argparse
from tqdm import tqdm


def main(task):
    # Paths
    image_dir = "smaller_images"
    labels_file = f"tasks/{task}/labels.txt"
    task_dir = f"tasks/{task}/smaller_images"

    # Ensure the removed_images directory exists
    os.makedirs(task_dir, exist_ok=True)

    # Load filenames from max_speed_labels.txt (without extensions)
    valid_files = set()
    with open(labels_file, "r") as file:
        for line in file:
            filename = line.split(":")[0]  # Get the filename (without extension)
            valid_files.add(filename)

    # Move .webp files not in max_speed_labels.txt to removed_images
    for file_name in tqdm(os.listdir(image_dir), f"{task}"):
        if file_name.endswith(".webp"):
            base_name = os.path.splitext(file_name)[0]
            if base_name in valid_files:
                source_path = os.path.join(image_dir, file_name)
                dest_path = os.path.join(task_dir, file_name)
                shutil.copy(source_path, dest_path)
    shutil.copy(labels_file, task_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="visualizer", description="Query GPT")
    parser.add_argument("task", type=str, help="task_to_create")
    args = parser.parse_args()
    main(args.task)
