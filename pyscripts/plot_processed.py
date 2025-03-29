import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import processed_tile_group_pb2 as proto_schema


def read_pbf_file(file_path):
    tile_group = proto_schema.TileGroup()
    with open(file_path, "rb") as f:
        tile_group.ParseFromString(f.read())
    return tile_group


def print_stats(tiles):
    print(f"Number of Tiles: {len(tiles)}")

    for tile in tiles:
        intra = zip(tile.intra_edges[0::2], tile.intra_edges[1::2])
        first_n = 0
        nbr = 0
        for n in tile.node_to_feature:
            if n != first_n:
                print(f"Feature({first_n}) has {nbr} nodes")
                first_n = n
                nbr = 0
            nbr += 1

        for i, (n1, n2) in enumerate(intra):
            f1 = tile.node_to_feature[n1]
            f2 = tile.node_to_feature[n2]
            print(f"Intra-Edge({i}): {n1} -> {n2}, Features({f1}, {f2})")
            if f1 != f2:
                print(
                    f"Warning: intra-edge connecting different features. Intra-Edge({i}): {n1} -> {n2}, Features({f1}, {f2})"
                )

    total_features = 0
    total_nodes = 0
    total_inter_edges = 0
    total_intra_edges = 0
    for tile in tiles:
        total_features += len(tile.features)
        total_nodes += len(tile.nodes) // 2  # Each node has lat and lon
        total_inter_edges += len(tile.inter_edges) // 2  # Each edge has n1 and n2
        total_intra_edges += len(tile.intra_edges) // 2

    print(f"Total Features: {total_features}")
    print(f"Total Nodes: {total_nodes}")
    print(f"Total Inter-Edges: {total_inter_edges}")
    print(f"Total Intra-Edges: {total_intra_edges}")

    # Additional stats: Feature tags and bounding boxes
    total_min_boxes = 0
    tag_count = dict()
    for tile in tiles:
        print(f"Tile: zoom={tile.zoom}, x={tile.x}, y={tile.y}")
        print(f"Feature Count: {len(tile.features)}")
        for feature in tile.features:
            total_min_boxes += 1
            tags = dict(zip(feature.tags[::2], feature.tags[1::2]))
            for k, v in tags.items():
                tag_count[(k, v)] = tag_count.get((k, v), 0) + 1

    tag_counts = sorted(tag_count.items(), key=lambda x: (x[1], x[0]), reverse=False)
    for kv in tag_counts:
        print(f"{str(kv[0]).ljust(60, '.')}{kv[1]}")
    print(f"Number of Tiles: {len(tiles)}")
    print(f"Total Min Boxes: {total_min_boxes}")


def generate_lat_lon_array(F=100):
    # Define the base lat and lon values (start point)
    base_lat = 0.0
    base_lon = 0.0

    # Define the box size and separation
    box_size = 0.2
    lon_separation = 0.2
    # Generate 4 lat/lon points for each feature
    lat_points = np.array(
        [base_lat, base_lat, base_lat + box_size, base_lat + box_size]
    )
    lon_points = np.array(
        [base_lon, base_lon + lon_separation, base_lon, base_lon + lon_separation]
    )
    # Combine them into a 4x2 array (4 lat/lon pairs for each feature)
    lat_lon_template = np.column_stack((lat_points, lon_points))
    # Repeat the lat/lon pairs F times
    lat_lon_array = np.tile(lat_lon_template, (F, 1, 1))
    return lat_lon_array


def plot_stuff(args, tile, ax, min_x, min_y, local_coords=False, dx=0.0, dy=0.0):
    node_to_feature = np.array(tile.node_to_feature, dtype=int)
    inter_edges = np.array(tile.inter_edges, dtype=int)
    intra_edges = np.array(tile.intra_edges, dtype=int)

    edges_inside = np.stack((inter_edges[::2], inter_edges[1::2]))
    edges_apart = np.stack((intra_edges[::2], intra_edges[1::2]))

    dx = tile.x - min_x + dx
    dy = tile.y - min_y + dy
    nodes = np.array(tile.nodes, dtype=float)
    nodes = nodes.reshape(-1, 2)
    if local_coords:
        nodes = np.array(tile.local_coords, dtype=float)
        print(nodes.max())
        print(nodes.min())
        nodes *= 0.2
        nodes = nodes.reshape(-1, 2)
        offset = np.column_stack((np.zeros(len(nodes)), node_to_feature * 0.2))
        nodes += offset
        dx *= 0.2
        dy *= 0.2

    # only for local_coords
    boxes = generate_lat_lon_array(len(tile.features))

    nodes += np.array([-dy * 1.05, dx * 1.05])
    # Plot edges_inside
    points = []
    feats_found = []
    feat = tile.features[tile.node_to_feature[edges_inside[0, 0]]]
    for i in range(edges_inside.shape[1]):
        src, tgt = edges_inside[:, i]
        if feat.is_relation:
            ax.plot(
                nodes[[src, tgt], 1],
                nodes[[src, tgt], 0],
                markersize=5,
                color="blue",
                alpha=0.5,
                label="Relation",
            )
        if src not in points and tgt not in points or i == edges_inside.shape[1] - 1:
            points.sort()
            lons = nodes[points, 1].tolist()
            lats = nodes[points, 0].tolist()
            feats_found.append(feat)
            if not args.only_relations:
                if feat.is_point:
                    ax.plot(lons, lats, "bo", markersize=5, label="Point")
                elif feat.is_polyline and not args.only_points:
                    ax.plot(lons, lats, color="orange", linewidth=2, label="Polyline")
                elif feat.is_polygon and len(points) > 1 and not args.only_points:
                    ax.fill(
                        lons,
                        lats,
                        color="blue",
                        edgecolor="black",
                        alpha=0.5,
                        label="Polygon",
                    )

            # Plot features' bounding boxes
            if not args.only_points:
                if not local_coords:
                    if args.min_boxes:
                        ys, xs = feat.min_box[::2], feat.min_box[1::2]
                        if len(xs) > 0 and len(ys) > 0:
                            xs.append(xs[0])
                            ys.append(ys[0])
                            ax.plot(
                                np.array(xs) + dx * 1.05,
                                np.array(ys) - dy * 1.05,
                                "green",
                                linewidth=0.5,
                            )
                else:
                    ys = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
                    xs = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
                    for box_i in range(len(tile.features)):
                        ax.plot(
                            xs * 0.2 + 0.2 * box_i,
                            ys * 0.2 - 0.2,
                            "green",
                            linewidth=0.5,
                        )

            points = []
        feat = tile.features[tile.node_to_feature[src]]
        points.append(src)
        points.append(tgt)
    # print(len(feats_found), "/", len(tile.features))
    # for f in tile.features:
    #     if f not in feats_found:
    #         print(f)

    # Plot edges_apart
    if not args.only_points:
        for i in range(edges_apart.shape[1]):
            src, tgt = edges_apart[:, i]
            if tile.node_to_feature[src] != tile.node_to_feature[tgt]:
                ax.plot(
                    nodes[[src, tgt], 1],
                    nodes[[src, tgt], 0],
                    color="purple",
                    linewidth=4,
                    label="Edges Apart",
                )
            else:
                x_values = [nodes[src, 1], nodes[tgt, 1]]
                y_values = [nodes[src, 0], nodes[tgt, 0]]
                ax.plot(
                    x_values,
                    y_values,
                    color="red",
                    linewidth=1.0,
                    alpha=0.8,
                    linestyle="--",
                    label="Visibility Edges" if i == 0 else "",
                )

    # Plot nodes
    for i, (lat, lon) in enumerate(nodes):
        if i in np.unique(np.concatenate((inter_edges, intra_edges))):
            continue
        if (
            args.only_relations
            and not tile.features[tile.node_to_feature[i]].is_relation
        ):
            continue
        ax.plot(lon, lat, "ko", markersize=5, alpha=0.7, label="Nodes")
        if args.nodes:
            # Add node number
            ax.text(
                lon + 0.005 * (random.random() * 2 - 1),
                lat + 0.005 * (random.random() * 2 - 1),
                str(i),
                fontsize=10,
                color="black",
            )

        if args.min_boxes:
            feat = tile.features[tile.node_to_feature[i]]
            ys, xs = feat.min_box[::2], feat.min_box[1::2]
            if len(xs) > 0 and len(ys) > 0:
                xs.append(xs[0])
                ys.append(ys[0])
                ax.plot(
                    np.array(xs) + dx * 1.05,
                    np.array(ys) - dy * 1.05,
                    "green",
                    linewidth=0.5,
                )


def plot_graph_multipolygon(args, tiles):
    """Plots a graph/multipolygon based on edges and node coordinates.

    Parameters:
    - edges_inside (torch.Tensor or np.ndarray): Tensor of shape [2, N_inside]
      representing edges within polygons.
    - edges_apart (torch.Tensor or np.ndarray): Tensor of shape [2, N_apart]
      representing edges connecting different polygons or disjoint parts.
    - nodes (torch.Tensor or np.ndarray): Tensor of shape [num_nodes, 2]
      representing node coordinates (x, y).
    """

    # Initialize a matplotlib figure
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    min_x = min([tile.x for tile in tiles])
    min_y = min([tile.y for tile in tiles])
    for tile in tiles:
        plot_stuff(args, tile, ax, min_x, min_y, local_coords=False)
        if args.local_coords:
            plot_stuff(args, tile, ax, min_x, min_y, local_coords=True, dx=1.0, dy=1.0)
            break
    # Set equal aspect ratio to maintain spatial accuracy
    ax.set_aspect("equal")

    # Create custom legend to avoid duplicate labels
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="blue", lw=2, label="Feature edges"),
        Line2D([0], [0], color="red", lw=1, linestyle="--", label="Relation edges"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Nodes",
            markerfacecolor="black",
            markersize=5,
        ),
    ]
    ax.legend(handles=legend_elements, loc="best")

    # Set plot titles and labels
    ax.set_title("Graph / Multipolygon Visualization", fontsize=16)
    ax.set_xlabel("X Coordinate", fontsize=14)
    ax.set_ylabel("Y Coordinate", fontsize=14)

    # Optional: Add grid
    ax.grid(True, linestyle="--", alpha=0.5)

    # Show the plot
    plt.show()


def main(args, pbf_file_path):
    if not os.path.isfile(pbf_file_path):
        print(f"File {pbf_file_path} does not exist.")
        sys.exit(1)

    print(f"Reading PBF file: {pbf_file_path}")
    tile_group = read_pbf_file(pbf_file_path)

    tiles = tile_group.tiles

    if args.subtile is not None:
        print(f"Subtile: {args.subtile}")
        sc = list(map(int, args.subtile.split("_")))
        for tile in tile_group.tiles:
            if tile.x == sc[1] and tile.y == sc[2]:
                tiles = [tile]
                break

    print("\n--- Statistics ---")
    print_stats(tiles)

    print("\n--- Plotting Geometries ---")
    plot_graph_multipolygon(args, tiles)
    # plot_geometries(tile_group)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="visualizer", description="Process PBF file")
    parser.add_argument(
        "--only-relations", action="store_true", help="Only process relation parts"
    )
    parser.add_argument(
        "--only-points", action="store_true", help="Only process points"
    )
    parser.add_argument("--min-boxes", action="store_true", help="Show bounding boxes")
    parser.add_argument(
        "--local-coords",
        action="store_true",
        help="Display in min_box local coordinates",
    )
    parser.add_argument(
        "-i", "--input_file", type=str, help="Path to the input PBF file"
    )
    parser.add_argument("--subtile", type=str, default=None, help="Subtile")
    parser.add_argument("--nodes", action="store_true", help="Node numbers")
    args = parser.parse_args()
    main(args, args.input_file)
