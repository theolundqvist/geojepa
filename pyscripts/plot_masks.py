import time
from random import random

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.modules.tokenizer import Modality

import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon

import networkx as nx

from src.data.components.raw_tile_dataset import RawTileDataset
from src.data.components.tiles import uncollate_first_tile, Tile, collate_tiles
from src.modules.mock_tokenizer import MockTokenizer
from src.modules.masks import (
    RandomMask,
    ContiguousMask,
    ModalityMask,
    AreaMask,
)  # Ensure these are correctly imported
from src.utils.sort_utils import restore_tensor_order


def plot_masks(
    strategy_name, strat, tile_batch, tokens, positions, modalities, indices, idx
):
    start = time.time()
    ctx, tgt = strat(tokens, positions, modalities)
    print(f"{strategy_name} took {(time.time() - start) * 1e3:.2f} ms to mask")

    # Step 4: Plot the results
    sample_idx = idx  # Choose a sample to visualize

    image = (
        tile_batch.SAT_imgs[sample_idx].permute(1, 2, 0).cpu().numpy()
    )  # Convert to HWC format

    tile = uncollate_first_tile(tile_batch)
    ctx_mask_sample = ctx[sample_idx].bool()
    tgt_masks_sample = tgt[:, sample_idx, :].bool()
    positions_sample = positions[sample_idx]
    modalities_sample = modalities[sample_idx]
    indices = indices[sample_idx]

    num_targets = tgt_masks_sample.shape[0]
    ncols = num_targets + 2  # Original image + context + targets

    colors = ["red", "green", "yellow", "cyan", "magenta"]

    fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(ncols * 6, 12))
    if ncols == 1:
        axes = axes.reshape(
            2, 1
        )  # Ensure axes is a 2D array even if there's only one target

    # First row: Image plots (without bounding boxes)
    # First column: Original image
    axes[0, 0].imshow(image)
    axes[0, 0].plot(
        [0, 0, 224, 224, 0], [0, 224, 224, 0, 0], color="black", linewidth=1
    )
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Original")

    # Prepare masks
    image_tokens_mask = modalities_sample == Modality.IMG
    image_tokens_mask[torch.nonzero(image_tokens_mask)[0]] = False
    geometry_tokens_mask = (
        modalities_sample == Modality.OSM
    )  # | (modalities_sample == 2)

    # Second column: Context Image Tokens
    context_image_mask = ctx_mask_sample & image_tokens_mask
    excluded_image_mask = (~ctx_mask_sample) & image_tokens_mask

    # Create masked images for context
    context_image = create_masked_image(image, positions_sample, context_image_mask)
    axes[0, 1].imshow(context_image)
    axes[0, 1].plot(
        [0, 0, 224, 224, 0], [0, 224, 224, 0, 0], color="black", linewidth=1
    )
    axes[0, 1].axis("off")
    axes[0, 1].set_title("Context")

    # Remaining columns: Target Image Tokens
    for i in range(num_targets):
        tgt_mask_i = tgt_masks_sample[i].bool()

        # Masks for target, context, and excluded tokens
        target_image_mask = tgt_mask_i & image_tokens_mask
        context_image_mask_i = ctx_mask_sample & image_tokens_mask

        # Create masked images for targets
        target_image = create_target_image(
            image,
            positions_sample,
            target_image_mask,
            context_image_mask_i,
            colors[i % len(colors)],
        )
        axes[0, 2 + i].imshow(target_image)
        axes[0, 2 + i].plot(
            [0, 0, 224, 224, 0], [0, 224, 224, 0, 0], color="black", linewidth=1
        )
        axes[0, 2 + i].axis("off")
        axes[0, 2 + i].set_title(f"Target {i + 1}")

    # Second row: Geometry plots (without background image)
    # First column: Original Geometry Tokens
    axes[1, 0].axis("off")
    axes[1, 0].set_title("Original")
    for ax in axes[1, :]:
        ax.set_aspect(1)
    plot_geo(axes[1, 0], tile, geometry_tokens_mask, indices)

    # Second column: Context Geometry Tokens
    context_geometry_mask = ctx_mask_sample & geometry_tokens_mask
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Context")
    plot_geo(axes[1, 1], tile, context_geometry_mask, indices)

    # Remaining columns: Target Geometry Tokens
    for i in range(num_targets):
        tgt_mask_i = tgt_masks_sample[i].bool()
        target_geometry_mask = tgt_mask_i & geometry_tokens_mask

        axes[1, 2 + i].axis("off")
        axes[1, 2 + i].set_title(f"Target {i + 1}")

        # Plot context geometry tokens in light gray
        # plot_geo(axes[1, 2 + i], tile, context_geometry_mask_i)
        # Plot target geometry tokens with specified color
        plot_geo(axes[1, 2 + i], tile, target_geometry_mask, indices)

    plt.suptitle(f"{strategy_name} Strategy")
    plt.tight_layout()
    plt.show()


def plot_geo(ax, tile: Tile, feature_mask, indices):
    """
    Plots the feature graphs in a tile, considering only the features specified in feature_mask.
    A feature can be a point, polyline, or polygon.

    Args:
        tile: An object containing the tile data with attributes:
            - nodes: torch.Tensor of shape [N, 2], node coordinates (latitude, longitude)
            - inter_edges: torch.Tensor of shape [2, E_inter], edges connecting nodes within features
            - node_to_feature: torch.Tensor of shape [N], mapping from node index to feature index
        ax: Matplotlib Axes object where the features will be plotted.
        feature_mask: torch.Tensor containing the indices of features to plot.
    """
    # Convert tensors to NumPy arrays
    ax.set_aspect(1)
    ax.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], color="black", linewidth=1)
    nodes = tile.nodes.numpy()  # Shape: [N, 2]
    inter_edges = tile.inter_edges.numpy()  # Shape: [2, E_inter]
    node_to_feature = tile.node_to_feature.numpy()  # Shape: [N]
    mask = restore_tensor_order(feature_mask.unsqueeze(0), indices.unsqueeze(0))[0]
    if mask.nonzero().numel() == 0:
        return

    for feature_index in range(tile.nbr_features):
        if not mask[feature_index].item():
            continue
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


def create_masked_image(image, positions_sample, include_mask):
    H, W, C = image.shape
    masked_image = np.full_like(image, 128)  # Start with gray image
    mask = np.zeros((H, W), dtype=bool)

    # Include tokens
    for idx in torch.nonzero(include_mask, as_tuple=False).squeeze():
        bbox = positions_sample[idx]
        bbox = bbox.view(4, 2)
        if torch.abs(torch.max(bbox[:, 0]) - torch.min(bbox[:, 0])) > 0.9:
            continue
        corners = (bbox.cpu().numpy() * [W, H]).astype(np.int32)
        corners[:, 1] = H - corners[:, 1]
        rr, cc = polygon(corners[:, 1], corners[:, 0], shape=(H, W))
        mask[rr, cc] = True

    masked_image[mask] = image[mask]
    return masked_image


def create_target_image(image, positions_sample, target_mask, context_mask, color):
    H, W, C = image.shape
    target_image = np.full_like(image, 128)  # Start with gray image

    # # Context tokens with 50% opacity
    # mask_context = np.zeros((H, W), dtype=bool)
    # for idx in torch.nonzero(context_mask, as_tuple=False).squeeze():
    #     bbox = positions_sample[idx]
    #     corners = (bbox.view(4, 2).cpu().numpy() * [W, H]).astype(np.int32)
    #     rr, cc = polygon(corners[:, 1], corners[:, 0], shape=(H, W))
    #     mask_context[rr, cc] = True
    #
    # target_image[mask_context] = (image[mask_context] * 0.5 + 128 * 0.5).astype(np.uint8)

    # Target tokens with full opacity
    mask_target = np.zeros((H, W), dtype=bool)
    for idx in torch.nonzero(target_mask, as_tuple=False).squeeze():
        bbox = positions_sample[idx]
        bbox = bbox.view(4, 2)
        if torch.abs(torch.max(bbox[:, 0]) - torch.min(bbox[:, 0])) > 0.9:
            continue
        corners = (bbox.cpu().numpy() * [W, H]).astype(np.int32)
        corners[:, 1] = H - corners[:, 1]
        rr, cc = polygon(corners[:, 1], corners[:, 0], shape=(H, W))
        mask_target[rr, cc] = True

    target_image[mask_target] = image[mask_target]

    return target_image


def main():
    # Step 1: Get a TileBatch from your dataset
    data = RawTileDataset("data/tiles/huge/tasks/pretraining/val", split="")
    i = int(random() * 300)
    for i in range(i, i + 1000):
        tile_batch = data.__getitem__(i)
        if tile_batch[0].nbr_features > 50:
            break
        i += 1
    idx = 0
    tile_batch = collate_tiles([tile_batch[0]])
    print("plotting:", tile_batch.names()[idx])
    # Repeat next(iter_loader) as necessary to get the desired batch

    # Step 2: Create a MockTokenizer and process the TileBatch
    tokenizer = MockTokenizer(token_dim=1, tokenize_geometry=True, sort_spatially=True)
    start = time.time()
    tokens, positions, modalities = tokenizer(tile_batch)
    print(f"tokenized in {time.time() - start}s")

    # Step 3: Run masking strategies
    strats = [
        AreaMask(
            target_size=0.35, num_targets=4, min_context=0.10, min_ar=0.5, max_ar=2.0
        ),
        RandomMask(target_size=0.40, num_targets=4, min_context=0.1),
        ContiguousMask(target_size=0.35, num_targets=4, min_context=0.15),
        ModalityMask(min_context=0.1),
    ]

    for strat in strats:
        plot_masks(
            type(strat).__name__,
            strat,
            tile_batch,
            tokens,
            positions,
            modalities,
            tokenizer.indices,
            idx=idx,
        )


if __name__ == "__main__":
    main()
