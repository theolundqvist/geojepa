import argparse
import os
import shutil


def process_directories(directories, outdir):
    """
    Process files in multiple directories:
    - Identify duplicate files (by name) across all directories.
    - Save unique files to outdir.

    Parameters:
    directories (list): List of directory paths to process.
    outdir (str): Path to the output directory.

    Returns:
    list: A list of file names that are duplicates.
    """
    # Ensure outdir exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Track seen files and duplicates
    all_files = {}
    duplicates = set()

    # Process each directory
    for dir_path in directories:
        files = os.listdir(dir_path)
        for file in files:
            file_path = os.path.join(dir_path, file)
            if file in all_files:
                duplicates.add(file)
            else:
                all_files[file] = file_path

    # Copy unique files to outdir
    for file, file_path in all_files.items():
        if file not in duplicates:
            shutil.copy(file_path, os.path.join(outdir, file))

    return list(duplicates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine files from multiple directories and handle duplicates."
    )
    parser.add_argument(
        "-dirs",
        "--directories",
        nargs="+",
        type=str,
        required=True,
        help="List of directories to process (space-separated).",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        required=True,
        help="Output directory to store unique files.",
    )
    args = parser.parse_args()

    duplicates = process_directories(args.directories, args.outdir)
    print("Duplicates:", duplicates)

# python script.py --directories dir1 dir2 dir3 --outdir output_directory
