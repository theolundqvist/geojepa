import os
import sys
import random

import matplotlib.pyplot as plt
import unprocessed_tile_group_pb2 as tile_proto


def read_pbf_file(file_path):
    tile_group = tile_proto.TileGroup()
    with open(file_path, "rb") as f:
        tile_group.ParseFromString(f.read())
    return tile_group


def print_stats(tiles):
    # print(f"TileGroup: zoom={tile_group.zoom}, x={tile_group.x}, y={tile_group.y}")
    tag_count = dict()
    for tile in tiles:
        print(f"Tile: zoom={tile.zoom}, x={tile.x}, y={tile.y}")
        print(f"Feature Count: {len(tile.features)}")
        for feature in tile.features:
            for k, v in feature.tags.items():
                tag_count[(k, v)] = tag_count.get((k, v), 0) + 1

    # print all data as is
    for tile in tiles:
        print(f"Tile: zoom={tile.zoom}, x={tile.x}, y={tile.y}")
        for gid, group in enumerate(tile.groups):
            print(f"Group({gid})")
            for k, v in group.tags.items():
                print(f"    {k}: {v}")
            for fids in group.feature_indices:
                feat = tile.features[fids]
                print(f"    Feature({fids})-Points({len(feat.geometry.points)})")
                for k, v in feat.tags.items():
                    print(f"        {k}: {v}")
        # nodes = zip(tile.nodes[0::2], tile.nodes[1::2])
        # for i, (lat, lon) in enumerate(nodes):
        #     print(f"Feat({tile.node_to_feature(i)})-Node({i})-({lat},{lon})")
        # edges = zip(tile.nodes[0::2], tile.nodes[1::2])

    tag_counts = sorted(tag_count.items(), key=lambda x: (x[1], x[0]), reverse=False)
    for kv in tag_counts:
        print(f"{str(kv[0]).ljust(60, '.')}{kv[1]}")
    print(f"Number of Tiles: {len(tiles)}")

    total_features = 0
    total_groups = 0
    for tile in tiles:
        total_features += len(tile.features)
        total_groups += len(tile.groups)

    print(f"Total Features: {total_features}")
    print(f"Total Groups: {total_groups}")

    # Additional stats
    feature_types = {"Point": 0, "Polyline": 0, "Polygon": 0}
    for tile in tiles:
        for feature in tile.features:
            num_points = len(feature.geometry.points)
            if num_points == 1:
                feature_types["Point"] += 1
            elif num_points > 1 and not feature.geometry.is_closed:
                feature_types["Polyline"] += 1
            elif num_points > 1 and feature.geometry.is_closed:
                feature_types["Polygon"] += 1
    print("Feature Types:")
    for k, v in feature_types.items():
        print(f"  {k}: {v}")

    print("Printing first point found")
    br = False
    for tile in tiles:
        for f in tile.features:
            print(f.geometry.points[0].lat, f.geometry.points[0].lon)
            br = True
            break
        if br:
            break


def plot_geometries(args, tiles, only_relations):
    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    min_x = min([tile.x for tile in tiles])
    min_y = min([tile.y for tile in tiles])

    # To avoid duplicate legend entries
    plotted_polygons = False
    plotted_holes = False
    plotted_polylines = False
    plotted_points = False
    plotted_edges = False

    for tile in tiles:
        groups = tile.groups
        relation_features = set()
        for g in groups:
            for fid in g.feature_indices:
                relation_features.add(fid)

        if len(tile.features) == 0:
            continue
        # Extract nodes as (lat, lon) pairs
        nodes = list(
            zip(
                tile.features,
            )
        )  # Not used in this context

        for i, feature in enumerate(tile.features):
            if only_relations:
                if i not in relation_features:
                    continue
            geometry = feature.geometry
            points = geometry.points
            lats = [p.lat for p in points]
            lons = [p.lon for p in points]

            if len(points) == 1:
                # Plot points as blue dots
                ax.plot(
                    lons,
                    lats,
                    "bo",
                    markersize=5,
                    label="Point" if not plotted_points else "",
                )
                if args.nodes:
                    for p in points:
                        ax.text(
                            p.lon + 0.00001 * (random.random() * 2 - 1),
                            p.lat + 0.00001 * (random.random() * 2 - 1),
                            str(i),
                            fontsize=10,
                            color="black",
                        )
                plotted_points = True
            elif len(points) > 1 and not geometry.is_closed:
                # Plot polylines in orange
                ax.plot(
                    lons,
                    lats,
                    color="orange",
                    linewidth=2,
                    label="Polyline" if not plotted_polylines else "",
                )
                plotted_polylines = True
            elif len(points) > 1 and geometry.is_closed:
                # Determine if the polygon is a hole or a main polygon
                is_hole = geometry.inner
                color = "lightgrey" if is_hole else "blue"
                label = "Hole" if is_hole else "Polygon"
                ax.fill(
                    lons,
                    lats,
                    color=color,
                    edgecolor="black",
                    alpha=0.5,
                    label=label
                    if not (is_hole and plotted_holes)
                    and not (not is_hole and plotted_polygons)
                    else "",
                )
                if is_hole:
                    plotted_holes = True
                else:
                    plotted_polygons = True

    # Create a legend without duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for handle, label in zip(handles, labels):
        if label not in by_label:
            by_label[label] = handle
    if by_label:
        ax.legend(by_label.values(), by_label.keys())

    ax.set_title("Geometries Plot")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", "box")  # Ensure equal aspect ratio
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()


def main(args, pbf_file_path, subtile, only_relations, output_dir="output_csv"):
    if not os.path.isfile(pbf_file_path):
        print(f"File {pbf_file_path} does not exist.")
        sys.exit(1)

    print(f"Reading PBF file: {pbf_file_path}")
    tile_group = read_pbf_file(pbf_file_path)

    tiles = tile_group.tiles

    if subtile is not None:
        print(f"Subtile: {subtile}")
        sc = list(map(int, subtile.split("_")))
        for tile in tile_group.tiles:
            if tile.x == sc[1] and tile.y == sc[2]:
                tiles = [tile]
                break

    print("\n--- Statistics ---")
    print_stats(tiles)

    print("\n--- Plotting Geometries ---")
    plot_geometries(args, tiles, only_relations)  # Set outline=True to draw the outline

    print("\n--- Exporting to CSV ---")
    # export_to_csv(tile_group, output_dir)
    print("\nProcessing completed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_pbf.py <path_to_pbf_file> [output_directory]")
        sys.exit(1)

    # output_directory = sys.argv[2] if len(sys.argv) > 2 else "output_csv"
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--file", type=str, required=True)
    parser.add_argument("--subtile", type=str, default=None, required=False)
    parser.add_argument("--only-relations", action="store_true", required=False)
    parser.add_argument("--nodes", action="store_true", required=False)
    args = parser.parse_args()
    main(args, args.file, args.subtile, args.only_relations)
