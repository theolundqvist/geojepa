import matplotlib.pyplot as plt
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import numpy as np
import networkx as nx

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Import your dataset and tile utilities
from src.data.components.raw_tile_dataset import RawTileDataset


def plot_tile_geometry(ax, tile):
    """
    Plots the geometry of a tile.

    Args:
        ax: Matplotlib Axes object where the features will be plotted.
        tile: An object containing the tile data with attributes:
            - nodes: torch.Tensor of shape [N, 2], node coordinates (latitude, longitude)
            - inter_edges: torch.Tensor of shape [2, E_inter], edges connecting nodes within features
            - node_to_feature: torch.Tensor of shape [N], mapping from node index to feature index
    """
    # Convert tensors to NumPy arrays
    nodes = tile.nodes.numpy()  # Shape: [N, 2]
    inter_edges = tile.inter_edges.numpy()  # Shape: [2, E_inter]
    node_to_feature = tile.node_to_feature.numpy()  # Shape: [N]
    ax.set_aspect(1)
    ax.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], color="black", linewidth=2)

    for feature_index in range(tile.nbr_features):
        # Find nodes belonging to the current feature
        nodes_in_feature = np.where(node_to_feature == feature_index)[0]
        if len(nodes_in_feature) == 0:
            continue  # Skip if no nodes are found for this feature

        # Find edges where both nodes belong to the current feature
        is_src_in_feature = node_to_feature[inter_edges[0]] == feature_index
        is_tgt_in_feature = node_to_feature[inter_edges[1]] == feature_index
        edges_mask = is_src_in_feature & is_tgt_in_feature
        edges_in_feature = inter_edges[:, edges_mask]

        # Build a graph for the current feature
        G = nx.Graph()
        G.add_nodes_from(nodes_in_feature)
        G.add_edges_from(edges_in_feature.T)  # Transpose to get list of (u, v)

        if len(nodes_in_feature) == 1:
            # Feature is a point
            node_idx = nodes_in_feature[0]
            lat, lon = nodes[node_idx]
            ax.plot(lon, lat, "bo", markersize=5)
        else:
            degrees = dict(G.degree())
            degree_values = list(degrees.values())

            if all(deg == 2 for deg in degree_values):
                # Feature is a polygon
                try:
                    cycle_edges = nx.find_cycle(G)
                    # Reconstruct the node sequence from the cycle edges
                    edge_dict = {}
                    for u, v in cycle_edges:
                        edge_dict.setdefault(u, []).append(v)
                        edge_dict.setdefault(v, []).append(u)
                    start_node = cycle_edges[0][0]
                    node_sequence = [start_node]
                    prev_node = None
                    current_node = start_node
                    while True:
                        neighbors = edge_dict[current_node]
                        next_node = (
                            neighbors[0] if neighbors[0] != prev_node else neighbors[1]
                        )
                        if next_node == start_node:
                            break
                        node_sequence.append(next_node)
                        prev_node, current_node = current_node, next_node
                    node_sequence.append(start_node)  # Close the polygon
                    coords = nodes[node_sequence]
                    lats = coords[:, 0]
                    lons = coords[:, 1]
                    ax.fill(lons, lats, color="blue", alpha=0.5, edgecolor="black")
                except nx.exception.NetworkXNoCycle:
                    # If no cycle is found, treat it as a polyline
                    ends = [node for node, deg in degrees.items() if deg == 1]
                    if ends:
                        start_node = ends[0]
                    else:
                        start_node = nodes_in_feature[0]
                    node_sequence = list(nx.dfs_preorder_nodes(G, start_node))
                    coords = nodes[node_sequence]
                    lats = coords[:, 0]
                    lons = coords[:, 1]
                    ax.plot(lons, lats, color="orange", linewidth=2)
            else:
                # Feature is a polyline
                ends = [node for node, deg in degrees.items() if deg == 1]
                if ends:
                    start_node = ends[0]
                else:
                    start_node = nodes_in_feature[0]
                node_sequence = list(nx.dfs_preorder_nodes(G, start_node))
                coords = nodes[node_sequence]
                lats = coords[:, 0]
                lons = coords[:, 1]
                ax.plot(lons, lats, color="orange", linewidth=2)


def main():
    # Load the dataset
    data = RawTileDataset("data/tiles/huge/processed", split="")

    # Get the first 16 tiles
    tiles = data.__getitem__(0)

    images = [tile.SAT_img for tile in tiles]  # Shape: [batch_size, C, H, W]

    # Set up the figure
    fig = plt.figure(figsize=(20, 10))

    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)

    gs_left = GridSpecFromSubplotSpec(
        4, 4, subplot_spec=gs[0], wspace=0.05, hspace=0.05
    )
    gs_right = GridSpecFromSubplotSpec(
        4, 4, subplot_spec=gs[1], wspace=0.05, hspace=0.05
    )

    axes_images = []
    axes_geometries = []

    for i in range(len(tiles)):
        row = i % 4
        col = i // 4
        ax_image = fig.add_subplot(gs_left[row, col])
        ax_geometry = fig.add_subplot(gs_right[row, col])
        axes_images.append(ax_image)
        axes_geometries.append(ax_geometry)

    # Plot images
    for i, ax in enumerate(axes_images):
        image = images[i].permute(1, 2, 0).numpy()
        ax.imshow(image)
        ax.axis("off")

    # Plot geometries
    for i, ax in enumerate(axes_geometries):
        tile = tiles[i]
        plot_tile_geometry(ax, tile)
        ax.axis("off")

    plt.suptitle("First 16 Tiles - Images and Geometries", fontsize=16)
    plt.show()


if __name__ == "__main__":
    main()
