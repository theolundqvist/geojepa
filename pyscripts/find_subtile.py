import glob
from typing import Tuple

import processed_tile_group_pb2 as pbf

from tqdm import tqdm


def get_parent_tile(new_x, new_y) -> Tuple[int, int]:
    zoom_diff = 16 - 14
    old_x = new_x // (2**zoom_diff)
    old_y = new_y // (2**zoom_diff)
    return old_x, old_y


def main(input_dir, subtile):
    pbf_files = glob.glob(f"{input_dir}/**/*.pbf", recursive=True)
    if not pbf_files:
        print(f"Warning: No .pbf files found in directory '{input_dir}'.")
    for file in tqdm(pbf_files, f"Processing task: {input_dir}"):
        tg = pbf.TileGroup()
        try:
            with open(file, "rb") as f:
                tg.ParseFromString(f.read())
        except Exception as e:
            print(f"Warning: Failed to parse '{f}': {e}")
            continue
        for tile in tg.tiles:
            tile_id = f"{tile.zoom}_{tile.x}_{tile.y}"
            if tile_id == subtile:
                print(f"Found tile: {tile_id} in file {file}")
                if get_parent_tile(tile.x, tile.y) != (tg.x, tg.y):
                    print(f"Invalid group for this tile: {tile_id} in file {file}")
                    print(f"Should be in group: 14_{get_parent_tile(tile.x, tile.y)}")
                    continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory containing pbf files.")
    parser.add_argument("--subtile", type=str, help="Name of subtile z_x_y")
    args = parser.parse_args()
    main(args.dir, args.subtile)
