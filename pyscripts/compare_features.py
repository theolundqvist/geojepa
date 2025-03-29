import argparse
import os
import processed_tile_group_pb2 as proto_schema
from collections import Counter


def read_pbf_file(file_path):
    tile_group = proto_schema.TileGroup()
    with open(file_path, "rb") as f:
        tile_group.ParseFromString(f.read())
    return tile_group


def get_feature_counts(tiles):
    feature_counter = Counter()
    for tile in tiles:
        for feature in tile.features:
            tags = tuple(
                zip(feature.tags[::2], feature.tags[1::2])
            )  # Convert tags to tuples
            feature_counter[tags] += 1
    return feature_counter


def print_stats(tiles):
    stats = []
    tags = []
    stats.append(f"Number of Tiles: {len(tiles)}")

    total_features = 0
    total_nodes = 0
    total_inter_edges = 0
    total_intra_edges = 0
    for tile in tiles:
        total_features += len(tile.features)
        total_nodes += len(tile.nodes) // 2  # Each node has lat and lon
        total_inter_edges += len(tile.inter_edges) // 2  # Each edge has n1 and n2
        total_intra_edges += len(tile.intra_edges) // 2

    stats.append(f"Total Features: {total_features}")
    stats.append(f"Total Nodes: {total_nodes}")
    stats.append(f"Total Inter-Edges: {total_inter_edges}")
    stats.append(f"Total Intra-Edges: {total_intra_edges}")

    # Additional stats: Feature tags and bounding boxes
    total_min_boxes = 0
    tag_count = dict()
    for tile in tiles:
        tags.append(f"Tile: zoom={tile.zoom}, x={tile.x}, y={tile.y}")
        tags.append(f"Feature Count: {len(tile.features)}")
        for feature in tile.features:
            total_min_boxes += 1
            tag_dict = dict(zip(feature.tags[::2], feature.tags[1::2]))
            for k, v in tag_dict.items():
                tag_count[(k, v)] = tag_count.get((k, v), 0) + 1

    tag_counts = sorted(tag_count.items(), key=lambda x: x[1], reverse=False)
    for key, value in tag_counts:
        tags.append(f"{str(key).ljust(60, '.')}{value}")
    stats.append(f"Number of Tiles: {len(tiles)}")
    stats.append(f"Total Min Boxes: {total_min_boxes}")
    return stats, tags


def compare_features(sub_path, z_x_y, subtile, print_tags, other_dir):
    # Corrected base directory calculation (reduce directory traversal)
    base_dir = os.path.dirname(sub_path)
    if not other_dir:
        other_dir = base_dir + "_cheat"
    subdir = os.path.basename(sub_path)

    file1 = os.path.join(base_dir, subdir, f"{z_x_y}.pbf")
    file2 = os.path.join(other_dir, subdir, f"{z_x_y}.pbf")

    if not os.path.isfile(file1) or not os.path.isfile(file2):
        print(f"One or both files ({file1}, {file2}) do not exist.")
        return

    # Read PBF files
    tile_group1 = read_pbf_file(file1)
    tile_group2 = read_pbf_file(file2)
    print(file1)
    print(file2)

    tiles1 = tile_group1.tiles
    tiles2 = tile_group2.tiles

    # Filter for subtile if provided
    if subtile:
        sc = list(map(int, subtile.split("_")))
        tiles1 = [tile for tile in tiles1 if tile.x == sc[1] and tile.y == sc[2]]
        tiles2 = [tile for tile in tiles2 if tile.x == sc[1] and tile.y == sc[2]]

    # Count features
    feature_counts1 = get_feature_counts(tiles1)
    feature_counts2 = get_feature_counts(tiles2)

    stats1, tags1 = print_stats(tiles1)
    stats2, tags2 = print_stats(tiles2)

    # print("\n".join(stats1))
    # print("\n")
    # print("\n".join(tags1))
    # print("\n")
    # print("\n".join(stats2))
    # print("\n")
    # print("\n".join(tags2))
    # print("\n")

    if print_tags:
        print("\n".join(stats2))
        print("\n".join(tags2))

    print("Differences in stats: ")
    differences_stats = sorted(list(set(stats1) ^ set(stats2)))
    print("\n".join(differences_stats))
    print("Differences in tags: ")
    differences_tags = sorted(list(set(tags1) ^ set(tags2)))
    print("\n".join(differences_tags))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare features and feature counts between a directory and its '_cheat' version."
    )
    parser.add_argument(
        "-i",
        "--sub_path",
        type=str,
        required=True,
        help="Subdirectory path (e.g., tasks/bridge_car/test)",
    )
    parser.add_argument(
        "-tile",
        "--z_x_y",
        type=str,
        required=True,
        help="The z_x_y PBF file name to compare (without path)",
    )
    parser.add_argument(
        "--subtile", type=str, default=None, help="Subtile to compare (format: 'z_x_y')"
    )
    parser.add_argument(
        "--print_tags",
        action="store_true",
        help="Optional flag to print all present args in cheat",
    )
    parser.add_argument(
        "--other_dir",
        default=None,
        help="Optional input to compare with other file than cheat",
    )
    args = parser.parse_args()
    compare_features(
        args.sub_path, args.z_x_y, args.subtile, args.print_tags, args.other_dir
    )
