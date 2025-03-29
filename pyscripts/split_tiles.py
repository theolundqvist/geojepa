import os
import random
import shutil
import argparse

from tqdm import tqdm  # Import tqdm for progress bars


def clear_directory(directory, extensions=None):
    """Remove all files from a directory with specified extensions."""
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                if extensions is None or file.endswith(extensions):
                    os.remove(file_path)


def read_existing_log(logfile_path):
    """Read the existing log file and return a dictionary mapping files to sets."""
    if not os.path.exists(logfile_path):
        return {}

    split_map = {}
    with open(logfile_path) as logfile:
        for line in logfile:
            file, set_name = line.strip().split(" -> ")
            split_map[file] = set_name
    return split_map


def write_to_log(logfile_path, file, split_name):
    """Append file and its assigned set to the log."""
    with open(logfile_path, "a") as logfile:
        logfile.write(f"{file} -> {split_name}\n")


def split_files_into_sets(
    merged_dir,
    logfile_path,
    base_dir,
    training_dir,
    validation_dir,
    testing_dir,
    seed=42,
):
    # Clear the output directories before copying .pbf and .webp files
    clear_directory(training_dir, (".pbf", ".webp"))
    clear_directory(validation_dir, (".pbf", ".webp"))
    clear_directory(testing_dir, (".pbf", ".webp"))

    # Ensure output directories exist
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(testing_dir, exist_ok=True)

    if logfile_path == None:
        # Path for log file
        logfile_path = os.path.join(base_dir, "file_split_log.txt")
    else:
        logfile_path = os.path.join(logfile_path, "file_split_log.txt")

    # Load the existing split from log if it exists
    split_map = read_existing_log(logfile_path)

    # Collect unique filenames (without extensions)
    file_pairs = {}
    for file in os.listdir(merged_dir):
        name, ext = os.path.splitext(file)
        if ext not in [".pbf", ".webp", ".png"]:
            continue
        if name not in file_pairs:
            file_pairs[name] = []
        file_pairs[name].append(file)

    # Initialize sets for training, validation, and testing
    training_set = []
    validation_set = []
    testing_set = []

    # Assign files to the respective sets based on the log
    for file in file_pairs.keys():
        if file in split_map:
            split_name = split_map[file]
            if split_name == "train":
                training_set.append(file)
            elif split_name == "val":
                validation_set.append(file)
            elif split_name == "test":
                testing_set.append(file)

    # Files to split that aren't in the log yet
    unsplit_files = [name for name in file_pairs.keys() if name not in split_map]

    # Set random seed for unsplit files
    random.seed(seed)
    random.shuffle(unsplit_files)

    # Define split ratios for new unsplit files
    total_files = len(unsplit_files)
    training_count = int(0.8 * total_files)
    validation_count = int(0.1 * total_files)

    # Split unsplit files into training, validation, and testing sets
    training_set += unsplit_files[:training_count]
    validation_set += unsplit_files[training_count : training_count + validation_count]
    testing_set += unsplit_files[training_count + validation_count :]

    # Function to copy files to target directory and log them
    def copy_files(file_list, target_dir, split_name):
        for name in tqdm(file_list, desc=f"Copying files to {split_name}", unit="file"):
            for file in file_pairs[name]:
                src = os.path.join(merged_dir, file)
                if os.path.isfile(src):
                    dst = os.path.join(target_dir, file)
                    shutil.copy2(src, dst)
                    # Log the split
                    name, ext = os.path.splitext(file)
                    if name not in split_map:
                        write_to_log(logfile_path, name, split_name)

    # Copy files to their respective directories and log
    copy_files(training_set, training_dir, "train")
    copy_files(validation_set, validation_dir, "val")
    copy_files(testing_set, testing_dir, "test")

    print(
        f"Split complete: {len(training_set)} in training, {len(validation_set)} in validation, {len(testing_set)} in testing."
    )
    print(f"Log file updated at {logfile_path}")


def run(merged_dir, logfile_path, base_dir):
    training_dir = base_dir + "/train"
    validation_dir = base_dir + "/val"
    testing_dir = base_dir + "/test"

    split_files_into_sets(
        merged_dir, logfile_path, base_dir, training_dir, validation_dir, testing_dir
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="pair_tiles",
        description="Combine satellite images and pbf tiles into training, validation, and test",
    )

    parser.add_argument(
        "-i",
        "--merged_dir",
        type=str,
        help="Directory containing merged satellite images and pbf files",
    )
    parser.add_argument(
        "-l",
        "--logfile_path",
        default=None,
        type=str,
        help="File containing split-info",
    )
    parser.add_argument(
        "-o", "--base_dir", type=str, help="Directory to put combined tiles"
    )

    args = parser.parse_args()

    os.makedirs(args.base_dir, exist_ok=True)
    run(args.merged_dir, args.logfile_path, args.base_dir)
