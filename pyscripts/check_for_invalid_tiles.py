import glob
from typing import Tuple

import src.data.components.processed_tile_group_pb2 as pbf

from tqdm import tqdm


def get_parent_tile(new_x, new_y) -> Tuple[int, int]:
    zoom_diff = 16 - 14
    old_x = new_x // (2**zoom_diff)
    old_y = new_y // (2**zoom_diff)
    return old_x, old_y


def main(input_dir):
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
            if get_parent_tile(tile.x, tile.y) != (tg.x, tg.y):
                print(f"Invalid tile: {tile_id} in file {file}")
                continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Directory containing pbf files.")
    args = parser.parse_args()
    main(args.input_dir)
