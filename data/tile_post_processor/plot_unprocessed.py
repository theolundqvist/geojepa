import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import unprocessed_tile_group_pb2 as tile_proto


def read_pbf_file(file_path):
    tile_group = tile_proto.TileGroup()
    with open(file_path, "rb") as f:
        tile_group.ParseFromString(f.read())
    return tile_group


def print_stats(tile_group):
    print(f"TileGroup: zoom={tile_group.zoom}, x={tile_group.x}, y={tile_group.y}")
    print(f"Number of Tiles: {len(tile_group.tiles)}")

    total_features = 0
    total_groups = 0
    for tile in tile_group.tiles:
        total_features += len(tile.features)
        total_groups += len(tile.groups)

    print(f"Total Features: {total_features}")
    print(f"Total Groups: {total_groups}")

    # Additional stats
    feature_types = {"Point": 0, "Polyline": 0, "Polygon": 0}
    for tile in tile_group.tiles:
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


def plot_geometries(tile_group):
    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    min_x = min([tile.x for tile in tile_group.tiles])
    min_y = min([tile.y for tile in tile_group.tiles])

    # To avoid duplicate legend entries
    plotted_polygons = False
    plotted_holes = False
    plotted_polylines = False
    plotted_points = False
    plotted_edges = False

    for tile in tile_group.tiles:
        if len(tile.features) == 0:
            continue
        # Extract nodes as (lat, lon) pairs
        nodes = list(
            zip(
                tile.features,
            )
        )  # Not used in this context

        for feature in tile.features:
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

    # # Plotting edges from groups
    # for tile in tile_group.tiles:
    #     for group in tile.groups:
    #         feature_indices = group.feature_indices
    #         if len(feature_indices) < 2:
    #             continue  # Not enough features to form an edge
    #         # Assuming each pair in feature_indices represents an edge
    #         for i in range(0, len(feature_indices), 2):
    #             if i + 1 >= len(feature_indices):
    #                 break  # Incomplete pair
    #             fid1, fid2 = feature_indices[i], feature_indices[i + 1]
    #             if fid1 >= len(tile.features) or fid2 >= len(tile.features):
    #                 continue  # Invalid feature index
    #             feature1 = tile.features[fid1]
    #             feature2 = tile.features[fid2]
    #             # Assuming features are points for edge plotting
    #             if len(feature1.geometry.points) != 1 or len(feature2.geometry.points) != 1:
    #                 continue  # Only plot edges between points
    #             lat1, lon1 = feature1.geometry.points[0].lat, feature1.geometry.points[0].lon
    #             lat2, lon2 = feature2.geometry.points[0].lat, feature2.geometry.points[0].lon
    #             ax.plot(
    #                 [lon1, lon2],
    #                 [lat1, lat2],
    #                 color="red",
    #                 linewidth=1,
    #                 label="Edge" if not plotted_edges else "",
    #             )
    #             plotted_edges = True

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


def export_to_csv(tile_group, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Export Features
    features_data = []
    for tile in tile_group.tiles:
        for idx, feature in enumerate(tile.features):
            feature_dict = {
                "tile_zoom": tile.zoom,
                "tile_x": tile.x,
                "tile_y": tile.y,
                "feature_index": idx,
                "tags": dict(feature.tags),
            }
            geometry = feature.geometry
            feature_dict["is_closed"] = geometry.is_closed
            feature_dict["inner"] = geometry.inner
            feature_dict["num_points"] = len(geometry.points)
            points = [(p.lat, p.lon) for p in geometry.points]
            feature_dict["points"] = points
            features_data.append(feature_dict)
    features_df = pd.DataFrame(features_data)
    features_df.to_csv(os.path.join(output_dir, "features.csv"), index=False)
    print(f"Exported Features to {os.path.join(output_dir, 'features.csv')}")

    # Export Groups
    groups_data = []
    for tile in tile_group.tiles:
        for group in tile.groups:
            group_dict = {
                "tile_zoom": tile.zoom,
                "tile_x": tile.x,
                "tile_y": tile.y,
                "tags": dict(group.tags),
                "feature_indices": list(group.feature_indices),
            }
            # Export edges as a list of tuples
            groups_data.append(group_dict)
    groups_df = pd.DataFrame(groups_data)
    groups_df.to_csv(os.path.join(output_dir, "groups.csv"), index=False)
    print(f"Exported Groups to {os.path.join(output_dir, 'groups.csv')}")

    # Export Tiles
    tiles_data = []
    for tile in tile_group.tiles:
        tile_dict = {
            "zoom": tile.zoom,
            "x": tile.x,
            "y": tile.y,
            "num_features": len(tile.features),
            "num_groups": len(tile.groups),
        }
        tiles_data.append(tile_dict)
    tiles_df = pd.DataFrame(tiles_data)
    tiles_df.to_csv(os.path.join(output_dir, "tiles.csv"), index=False)
    print(f"Exported Tiles to {os.path.join(output_dir, 'tiles.csv')}")

    # Export TileGroup
    tilegroup_dict = {
        "zoom": tile_group.zoom,
        "x": tile_group.x,
        "y": tile_group.y,
        "num_tiles": len(tile_group.tiles),
    }
    tilegroup_df = pd.DataFrame([tilegroup_dict])
    tilegroup_df.to_csv(os.path.join(output_dir, "tilegroup.csv"), index=False)
    print(f"Exported TileGroup to {os.path.join(output_dir, 'tilegroup.csv')}")


def main(pbf_file_path, output_dir="output_csv"):
    if not os.path.isfile(pbf_file_path):
        print(f"File {pbf_file_path} does not exist.")
        sys.exit(1)

    print(f"Reading PBF file: {pbf_file_path}")
    tile_group = read_pbf_file(pbf_file_path)

    print("\n--- Statistics ---")
    print_stats(tile_group)

    print("\n--- Plotting Geometries ---")
    plot_geometries(tile_group)  # Set outline=True to draw the outline

    print("\n--- Exporting to CSV ---")
    export_to_csv(tile_group, output_dir)
    print("\nProcessing completed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_pbf.py <path_to_pbf_file> [output_directory]")
        sys.exit(1)

    pbf_file = sys.argv[1]
    output_directory = sys.argv[2] if len(sys.argv) > 2 else "output_csv"
    main(pbf_file, output_directory)
