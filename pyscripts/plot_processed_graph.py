import argparse
import os
import sys
import logging

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import processed_tile_group_pb2 as proto_schema


def read_pbf_file(file_path):
    tile_group = proto_schema.TileGroup()
    with open(file_path, "rb") as f:
        tile_group.ParseFromString(f.read())
    return tile_group


def print_stats(tiles):
    print(f"Number of Tiles: {len(tiles)}")

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


def build_graph_from_tile(tile, args):
    G = nx.Graph()

    # Convert nodes array into (lat, lon) coordinate tuples
    lats = tile.nodes[::2]
    lons = tile.nodes[1::2]

    # Add nodes with positions
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        if (
            args.only_relations
            and not tile.features[tile.node_to_feature[i]].is_relation
        ):
            continue
        G.add_node(i, pos=(lon, lat), feature=tile.features[tile.node_to_feature[i]])

    # Convert edges (both inter and intra) into a list of node index pairs
    inter_edges = list(zip(tile.inter_edges[::2], tile.inter_edges[1::2]))
    intra_edges = list(zip(tile.intra_edges[::2], tile.intra_edges[1::2]))

    # Add edges to the graph only if both nodes exist
    for edge in inter_edges:
        if edge[0] in G.nodes and edge[1] in G.nodes:
            G.add_edge(*edge, type="inter")
        else:
            logging.debug(
                f"Skipping inter-edge {edge} because one or both nodes are missing."
            )

    for edge in intra_edges:
        if edge[0] in G.nodes and edge[1] in G.nodes:
            G.add_edge(*edge, type="intra")
        else:
            logging.debug(
                f"Skipping intra-edge {edge} because one or both nodes are missing."
            )

    return G


def draw_graph(G, ax, args):
    pos = nx.get_node_attributes(G, "pos")

    # Separate edges by type for styling
    inter_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "inter"]
    intra_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "intra"]
    print(inter_edges)

    # Draw inter-edges
    if inter_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=inter_edges,
            edge_color="blue",
            alpha=0.5,
            ax=ax,
            label="Inter-Edges",
        )

    # Draw intra-edges
    # if intra_edges:
    #     nx.draw_networkx_edges(G, pos, edgelist=intra_edges, edge_color='red', style='dashed', alpha=0.5, ax=ax, label='Intra-Edges')

    # Draw nodes
    if args.only_relations:
        node_color = "purple"
    elif args.only_points:
        node_color = "green"
    else:
        node_color = "black"
    nx.draw_networkx_nodes(
        G, pos, node_size=20, node_color=node_color, ax=ax, label="Nodes"
    )

    # Optionally, draw bounding boxes
    if args.min_boxes:
        for node, data in G.nodes(data=True):
            feature = data["feature"]
            if not feature.min_box:
                continue
            ys, xs = feature.min_box[::2], feature.min_box[1::2]
            if len(xs) > 0 and len(ys) > 0:
                xs = xs.copy()
                ys = ys.copy()
                xs.append(xs[0])
                ys.append(ys[0])
                ax.plot(
                    np.array(xs),
                    np.array(ys),
                    "green",
                    linewidth=0.5,
                    label="Bounding Box" if node == list(G.nodes)[0] else "",
                )


def plot_graph_multipolygon(args, tiles):
    """Plots a graph/multipolygon based on edges and node coordinates.

    Parameters:
    - tiles (list): List of tile objects.
    """

    # Initialize a matplotlib figure
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    min_x = min([tile.x for tile in tiles])
    min_y = min([tile.y for tile in tiles])

    for tile in tiles:
        G = build_graph_from_tile(tile, args)
        draw_graph(G, ax, args)
        if args.local_coords:
            # Handle local coordinates if needed
            # This part can be expanded based on how local_coords should affect the graph
            pass  # Placeholder for additional local coordinate handling

    # Set equal aspect ratio to maintain spatial accuracy
    ax.set_aspect("equal")

    # Create custom legend to avoid duplicate labels
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="blue", lw=2, label="Inter-Edges"),
        Line2D([0], [0], color="red", lw=1, linestyle="--", label="Intra-Edges"),
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

    if args.min_boxes:
        legend_elements.append(
            Line2D([0], [0], color="green", lw=1, label="Bounding Boxes")
        )

    ax.legend(handles=legend_elements, loc="best")

    # Set plot titles and labels
    ax.set_title("Graph / Multipolygon Visualization", fontsize=16)
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)

    # Optional: Add grid
    # ax.grid(True, linestyle="--", alpha=0.5)

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
        "-i", "--input_file", type=str, help="Path to the input PBF file", required=True
    )
    parser.add_argument("--subtile", type=str, default=None, help="Subtile")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)  # Change to DEBUG to see skipped edges

    main(args, args.input_file)
