import os


def run(dir):
    split_map = {}
    for split in ["train", "val", "test"]:
        files = os.listdir(f"{dir}/tasks/traffic_signals/{split}")
        for file in files:
            name, ext = os.path.splitext(file)
            if ext in [".pbf", ".webp"]:
                split_map[name] = split

    if len(split_map) == 0:
        print("No files found")
        return
    print(f"Total files to keep in split_log: {len(split_map)}")
    with open(f"{dir}/file_split_log.txt", "r") as f:
        print(f"Total length of split_log: {len(f.readlines())}")
    input("Press enter to replace split_log with these entries...")
    with open(f"{dir}/file_split_log.txt", "w") as logfile:
        for file, set_name in split_map.items():
            logfile.write(f"{file} -> {set_name}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean up split log file")
    parser.add_argument(
        "--dir", type=str, help="Directory path for split log file", required=True
    )
    args = parser.parse_args()
    run(args.dir)
