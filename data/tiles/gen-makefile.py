import argparse
import json
import os
import sys
from pathlib import Path

import requests

import platform


def get_os_name():
    """Determine the OS name."""
    os_name = platform.system().lower()

    if os_name.startswith("win"):
        return "win"
    elif os_name.startswith("darwin"):
        return "macos"
    elif os_name.startswith("linux"):
        return "linux"
    else:
        return "unknown_os"


def get_cpu_arch():
    """Determine the CPU architecture."""
    arch = platform.machine().lower()

    if arch in ["x86_64", "amd64", "x64"]:
        return "x64"
    elif arch in ["i386", "i686", "x86"]:
        return "x86"
    elif arch in ["aarch64", "armv8", "arm64"]:
        return "arm64"
    elif arch in ["arm", "armv7"]:
        return "arm"
    else:
        return "unknown_arch"


def generate_os_arch():
    """Generate a string representing the OS and architecture."""
    os_name = get_os_name()
    cpu_arch = get_cpu_arch()

    return f"{os_name}_{cpu_arch}"


def display_options(options):
    """Displays the download options to the user.

    Args:
        options (dict): Dictionary of options with keys as option numbers and values as option names.
    """
    print("Please select an option to download:")
    for key, value in options.items():
        print(f"{key}. {value}")
    print()


def get_user_choice(options):
    """Prompts the user to select an option.

    Args:
        options (dict): Dictionary of options.

    Returns:
        str: The selected option name.
    """
    while True:
        try:
            choice = int(input("Enter the number corresponding to your choice: "))
            if choice in options:
                return options[choice]
            else:
                print(
                    f"Invalid choice. Please select a number between {min(options.keys())} and {max(options.keys())}.\n"
                )
        except ValueError:
            print("Invalid input. Please enter a number.\n")


def download_file(url, dest_path, chunk_size=8192):
    """Downloads a file from a URL to a specified destination path.

    Args:
        url (str): The URL to download from.
        dest_path (Path): The destination file path.
        chunk_size (int): Size of each chunk to download. Default is 8KB.
    """
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Raise an error for bad status codes
            total_size = int(response.headers.get("content-length", 0))
            print(f"\nStarting download: {url}")
            print(f"Total size: {total_size / (1024 * 1024):.2f} MB\n")

            with open(dest_path, "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        downloaded += len(chunk)
                        done = int(50 * downloaded / total_size) if total_size else 0
                        done = min(done, 50)  # Ensure it doesn't exceed the bar
                        sys.stdout.write(
                            f"\r[{'█' * done}{'.' * (50 - done)}] "
                            f"{downloaded / (1024 * 1024):.2f} MB / {total_size / (1024 * 1024):.2f} MB"
                        )
                        sys.stdout.flush()
            print("\nDownload completed successfully.\n")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        sys.exit(1)


def make_scripts_executable(scripts):
    """Ensures that the provided scripts have execute permissions.

    Args:
        scripts (list of Path): List of script paths.
    """
    for script in scripts:
        if not script.is_file():
            print(f"Script '{script}' does not exist.")
            continue
        if not os.access(script, os.X_OK):
            print(f"Making script '{script}' executable.")
            try:
                script.chmod(script.stat().st_mode | 0o111)  # Add execute permissions
            except OSError as e:
                print(f"Failed to make '{script}' executable: {e}")
                sys.exit(1)


def main():
    # Define available options
    root = Path(__file__).resolve().parents[0]

    # Mapping of options to their download URLs and directories
    with open(root / "config.json") as f:
        config = json.load(f)
    options = {}
    for i, key in enumerate(config.keys(), 1):
        options[i] = key

    parser = argparse.ArgumentParser(
        description="Generate a Makefile for tiling and post-processing."
    )
    parser.add_argument(
        "--all", action="store_true", help="Generate Makefiles for all options."
    )
    args = parser.parse_args()

    # if args.all:
    if True:  # just always build all, because why not
        print("Generating Makefiles for all options.")
        for key in options.keys():
            selected_config = config.get(options[key])
            generate(root, selected_config)
        return
    # Display options and get user choice
    display_options(options)
    choice = get_user_choice(options)
    print(f"\nYou selected: {choice}\n")

    # Retrieve configuration based on choice
    selected_config = config.get(choice)
    if not selected_config:
        print("Configuration for the selected option is missing.")
        sys.exit(1)

    generate(root, selected_config)


def generate(root, config):
    download_url = config["url"]
    target_dir = root / Path(config["directory"])
    osm_filename = config["filename"]
    # Define the scripts to be used
    build = generate_os_arch()
    tiling_script = root / Path(f"../osm_tile_extractor/builds/{build}/tiling")
    post_process_script = root / Path(
        f"../tile_post_processor/builds/{build}/tile_post_processor"
    )
    task_generate_script = root / Path(
        f"../task_generator/builds/{build}/task_generator"
    )
    cooccurrence_script = root / Path(f"../task_generator/builds/{build}/cooccurrence")
    cooccurrence_script_python = root / Path("../../pyscripts/analyze_cooccurrence.py")
    split_images_script = root / Path(
        f"../sat_tile_downloader/builds/{build}/split_images"
    )
    build_h5_script = "pyscripts.build_h5_datasets"
    merge_images_script = root / Path(
        f"../sat_tile_downloader/builds/{build}/merge_tiles"
    )
    merging_script = root / Path("../../pyscripts/merge_tiles.py")
    splitting_script = root / Path("../../pyscripts/split_tiles.py")
    pruning_script = "pyscripts.prune_dataset"
    stats_script = "pyscripts.count_labels"
    visualize_processed_script = root / Path("../tile_post_processor/plot_processed.py")
    visualize_unprocessed_script = root / Path(
        "../tile_post_processor/plot_unprocessed.py"
    )
    # pbf1 = root/Path("../tile_post_processor/builds/common/processed_tile_group_pb2.py")
    # pbf2 = root/Path("../tile_post_processor/builds/common/unprocessed_tile_group_pb2.py")

    # Create target directory if it doesn't exist
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{target_dir}' is ready.\n")
    except OSError as e:
        print(f"Error creating directory '{target_dir}': {e}")
        sys.exit(1)

    # Define the full path for the downloaded file
    osm_file_path = target_dir / osm_filename

    # Download the file
    print("skipping osm file download")
    # if os.path.exists(osm_file_path):
    #     print(f"{osm_filename} already downloaded, skipping")
    # else:
    #     print(f"downloading osm.pbf file to {osm_file_path}")
    #     download_file(download_url, osm_file_path)

    # Ensure scripts are executable
    scripts = [tiling_script, post_process_script, task_generate_script]
    make_scripts_executable(scripts)

    # Define script paths within the target directory
    tiling_script_path = tiling_script
    post_process_script_path = post_process_script
    task_generate_script_path = task_generate_script

    # Verify that the tiling and post-process scripts exist in the target directory
    # if not tiling_script_path.exists():
    #     print(f"Tiling script '{tiling_script_path}' does not exist.")
    #     sys.exit(1)

    if not post_process_script_path.exists():
        print(f"Post-process script '{post_process_script_path}' does not exist.")
        sys.exit(1)

    if not task_generate_script_path.exists():
        print(f"Task-gen script '{task_generate_script_path}' does not exist.")
        sys.exit(1)

    # target_dir_abs = os.path.abspath(target_dir)+"/scripts"
    # os.makedirs(target_dir_abs, exist_ok=True)
    # Run the tiling script
    # shutil.copy(tiling_script_path, target_dir_abs)
    # shutil.copy(post_process_script_path, target_dir_abs)
    # shutil.copy(task_generate_script_path, target_dir_abs)
    # shutil.copy(visualize_processed_script, target_dir_abs)
    # shutil.copy(visualize_unprocessed_script, target_dir_abs)
    # shutil.copy(pbf1, target_dir_abs)
    # shutil.copy(pbf2, target_dir_abs)

    template_path = root / "makefile.template"
    makefile_path = target_dir / "makefile"

    # Read the template
    with open(template_path, "r") as template_file:
        makefile_content = template_file.read()

    # Prepare replacements
    replacements = {
        "{{TILING_SCRIPT}}": str(tiling_script),
        "{{OSM_FILENAME}}": osm_filename,
        "{{MERGING_SCRIPT}}": str(merging_script),
        "{{SPLITTING_SCRIPT}}": str(splitting_script),
        "{{PRUNING_SCRIPT}}": str(pruning_script),
        "{{STATS_SCRIPT}}": str(stats_script),
        "{{POST_PROCESS_SCRIPT}}": str(post_process_script),
        "{{TASK_GENERATE_SCRIPT}}": str(task_generate_script),
        "{{COOCCURRENCE_SCRIPT}}": str(cooccurrence_script),
        "{{COOCCURRENCE_SCRIPT_PYTHON}}": str(cooccurrence_script_python),
        "{{SPLIT_IMAGES_SCRIPT}}": str(split_images_script),
        "{{BUILD_H5_SCRIPT}}": str(build_h5_script),
        "{{S3_BUCKET}}": "s3://tlundqvist/geojepa",
        "{{DATASET_NAME}}": str(Path(config["directory"])),
        # Add other placeholders as needed
    }

    # Replace placeholders
    for placeholder, actual in replacements.items():
        makefile_content = makefile_content.replace(placeholder, actual)

    # Write the populated Makefile
    with open(makefile_path, "w") as makefile:
        makefile.write("""
# =============================================================================
# Automatically generated. Do not modify directly.
# Use the corresponding Python script to regenerate if necessary.
# =============================================================================
        """)
        makefile.write(makefile_content)
    # run_script(
    #     tiling_script_path, f"
    # )
    #
    # # Run the post-process script
    # run_script(
    #     post_process_script_path, f"-i{target_dir}/unprocessed/", f"-o{target_dir}/processed/"
    # )

    print("\n\n------- SUCCESS -------\n")
    print(
        f"\nRun the following command to generate tiles: \n\ncd {Path(__file__).relative_to(os.getcwd()) / target_dir}\nmake tiles\nmake graphs\n"
    )


if __name__ == "__main__":
    main()
