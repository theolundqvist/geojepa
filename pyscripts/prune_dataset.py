import glob
import os
import random
import pyscripts.processed_tile_group_pb2 as tile_proto

from tqdm import tqdm  # Import tqdm for progress bars


def read_pbf_file(file_path):
    tile_group = tile_proto.TileGroup()
    with open(file_path, "rb") as f:
        tile_group.ParseFromString(f.read())
    return tile_group


def filter_files(root_dir, prune):
    print(root_dir)
    print("Pruning PBF files")
    for file in tqdm(glob.glob(root_dir + "/**/*.pbf")):
        prune(file)


def parse_file(file, name=""):
    if not os.path.isfile(file):
        print(file, "will be created")
        return {}
        raise FileNotFoundError(f"{file} not found.")

    current_name = None
    dict = {}
    with open(file, "r") as file:
        for line in file:
            line = line.strip()
            if ":" not in line:
                current_name = line
            elif name == current_name or name == "":
                key, value = line.split(":")
                dict[key] = value
    return dict


def run(directory, config_dir, task_name):
    config_file = os.path.join(config_dir, "pruning_config.txt")
    if not os.path.isfile(config_file):
        with open(config_file, "w") as f:
            f.write(task_name + "\n")

    print(f"Pruning 'task_name': {task_name}")
    config = parse_file(config_file, task_name)
    config = {float(key): float(value) for key, value in config.items()}
    print("config:", config)

    log_file = os.path.join(config_dir, f"pruning_logs_{task_name}.txt")
    prune_log = parse_file(log_file)

    labels_path = os.path.join(directory, "labels.txt")
    labels = parse_file(labels_path)
    labels = {key: float(value) for key, value in labels.items()}
    labels_keep = {}
    labels_prune = {}

    for key, val in tqdm(
        labels.items(),
        f"Pruning {task_name}{'_cheat' if '_cheat' in directory else ''}",
    ):
        prev_ass = prune_log.get(key, False)
        if not prev_ass:
            prop = config.get(val, 1)
            rand = random.random()
            if rand < prop:
                labels_keep[key] = val
                prune_log[key] = "keep"
            else:
                labels_prune[key] = val
                prune_log[key] = "prune"
        elif prev_ass == "keep":
            labels_keep[key] = val
        elif prev_ass == "prune":
            labels_prune[key] = val
        else:
            raise BaseException("LabelError: This should never be reached")

    print(f"to_keep: {len(labels_keep)}")
    print(f"to_prune: {len(labels_prune)}")

    for key, val in prune_log.items():
        if key not in labels:
            labels_prune[key] = 0.0

    print(f"to_keep: {len(labels_keep)}")
    print(f"to_prune: {len(labels_prune)}")

    # if len(labels_prune) == 0:
    #     return
    # Call the function to match and copy files
    def prune(pbf_path):
        tile_group = read_pbf_file(pbf_path)
        new_tile_group = tile_proto.TileGroup()
        new_tile_group.zoom = tile_group.zoom
        new_tile_group.x = tile_group.x
        new_tile_group.y = tile_group.y
        write = False
        for tile in tile_group.tiles:
            tilename = f"{tile.zoom}_{tile.x}_{tile.y}"
            if tilename in labels_keep:
                new_tile = new_tile_group.tiles.add()
                new_tile.CopyFrom(tile)
            else:
                write = True

        if len(new_tile_group.tiles) > 0:
            if write:
                with open(pbf_path, "wb") as f:
                    f.write(new_tile_group.SerializeToString())
        else:
            os.remove(pbf_path)

    if len(labels_prune) != 0:
        filter_files(directory, prune)

    with open(labels_path, "w") as f:
        for key, value in labels_keep.items():
            f.write(f"{key}:{value}\n")

    with open(log_file, "w") as f:
        for key, value in prune_log.items():
            f.write(f"{key}:{value}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="filter_dataset",
        description="Filter dataset into desired proportion according to config file",
    )

    parser.add_argument(
        "-i", "--dir", type=str, help="Directory containing files to filter"
    )
    parser.add_argument(
        "-d", "--config_dir", type=str, help="Config directory for filtering"
    )
    parser.add_argument(
        "-n", "--task_name", type=str, help="Task name, also file identifier"
    )

    args = parser.parse_args()

    run(args.dir, args.config_dir, args.task_name)
