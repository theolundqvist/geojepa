from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from torch import Tensor


def batch_geometries(
    node_list: List[Tensor],
    bbox_local_coords_list: List[Tensor],
    intra_edge_list: List[Tensor],
    inter_edge_list: List[Tensor],
    node_to_feature_list: List[Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""
    Args:
        node_list: list of geometry node tensors [[N, 2], ...]
        bbox_local_coords_list:
        intra_edge_list: list of intra edge tensors [[2, E_intra], ...]
        inter_edge_list: list of inter edge tensors [[2, E_inter], ...]
        node_to_feature_list: list of node to feature map tensors [[N], ...]

    Returns:
        nodes: [N_tot, 2]
        bbox_local_coords: [N_tot, 2]
        intra_edges: [2, E_intra_tot]
        inter_edges: [2, E_inter_tot]
        node_to_feature: [N_tot]
    """
    node_counts = torch.tensor(
        [n.size(0) for n in node_list], device=node_list[0].device
    )
    node_offsets = torch.cat(
        [
            torch.tensor([0], device=node_list[0].device),
            torch.cumsum(node_counts, dim=0)[:-1],
        ]
    )
    feat_counts = torch.tensor(
        [0 if len(n) == 0 else n[-1] + 1 for n in node_to_feature_list],
        device=node_list[0].device,
    )
    feature_offsets = torch.cat(
        [
            torch.tensor([0], device=node_list[0].device),
            torch.cumsum(feat_counts, dim=0)[:-1],
        ]
    )

    def cat_with_offset(tensor_list, shift_offsets):
        cat = torch.cat(tensor_list, dim=-1)
        counts = torch.tensor([e.size(-1) for e in tensor_list], device=cat.device)
        shift_repeated = shift_offsets.repeat_interleave(counts)
        return cat + shift_repeated.unsqueeze(0)

    # Adjust intra_edges and inter_edges
    node_batch = torch.cat(node_list, dim=0)
    local_coords_batch = torch.cat(bbox_local_coords_list, dim=0)
    intra_edges_batch = cat_with_offset(intra_edge_list, node_offsets)
    inter_edges_batch = cat_with_offset(inter_edge_list, node_offsets)
    node_to_feature_batch = cat_with_offset(
        node_to_feature_list, feature_offsets
    ).flatten()

    return (
        node_batch,
        local_coords_batch,
        intra_edges_batch,
        inter_edges_batch,
        node_to_feature_batch,
    )


## Below functions are adapted from torch_scatter and torch_geometric
def get_angle(v1, v2):
    if v1.shape[1] == 2:
        v1 = F.pad(v1, (0, 1), value=0)
    if v2.shape[1] == 2:
        v2 = F.pad(v2, (0, 1), value=0)
    return torch.atan2(
        torch.cross(v1, v2, dim=1).norm(p=2, dim=1), (v1 * v2).sum(dim=1)
    )


def scipy_spanning_tree(edge_index, edge_weight, num_nodes):
    row, col = edge_index.cpu()
    edge_weight = edge_weight.cpu()
    cgraph = csr_matrix((edge_weight, (row, col)), shape=(num_nodes, num_nodes))
    Tcsr = minimum_spanning_tree(cgraph)
    tree_row, tree_col = Tcsr.nonzero()
    spanning_edges = np.stack([tree_row, tree_col], 0)
    return spanning_edges


def build_spanning_tree_edge(edge_index, edge_weight, num_nodes):
    spanning_edges = scipy_spanning_tree(
        edge_index,
        edge_weight,
        num_nodes,
    )

    spanning_edges = torch.tensor(
        spanning_edges, dtype=torch.long, device=edge_index.device
    )
    spanning_edges_undirected = torch.cat(
        [spanning_edges, torch.stack([spanning_edges[1], spanning_edges[0]])], 1
    )
    return spanning_edges_undirected


def triplets(edge_index, num_nodes):
    row, col = edge_index
    device = row.device

    value = torch.arange(row.size(0), device=device)
    # Sort edges by source node
    sorted_idx = torch.argsort(row)
    row = row[sorted_idx]
    col = col[sorted_idx]
    value = value[sorted_idx]

    # Compute counts and offsets for source nodes
    counts = torch.bincount(row, minlength=num_nodes)
    offsets = torch.cat([torch.tensor([0], device=device), counts.cumsum(dim=0)[:-1]])

    # For each edge (i -> j), find edges (j -> k)
    edge_idx = torch.arange(row.size(0), device=device)
    j_counts = counts[col]
    j_offsets = offsets[col]

    num_triplets = j_counts

    # Repeat indices according to the number of triplets
    idx_e1 = edge_idx.repeat_interleave(num_triplets)
    idx_i = row[idx_e1]
    idx_j = col[idx_e1]

    # Generate indices for edges (j -> k)
    max_count = counts.max().item()
    arange_counts = torch.arange(max_count, device=device)
    mask = arange_counts.unsqueeze(0) < j_counts.unsqueeze(1)
    idx_e2 = (j_offsets.unsqueeze(1) + arange_counts.unsqueeze(0))[mask]
    idx_k = col[idx_e2]

    edx_1st = value[idx_e2]
    edx_2nd = value[idx_e1]

    # Apply masks to filter out unwanted triplets
    mask1 = (idx_i == idx_k) & (idx_j != idx_i)  # Remove go-back triplets
    mask2 = (idx_i == idx_j) & (idx_j != idx_k)  # Remove repeat self-loop triplets
    mask3 = (idx_j == idx_k) & (idx_i != idx_k)  # Remove self-loop neighbors
    mask = ~(mask1 | mask2 | mask3)

    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    idx_k = idx_k[mask]
    edx_1st = edx_1st[mask]
    edx_2nd = edx_2nd[mask]
    idx_e1 = idx_e1[mask]

    # Recalculate the number of triplets after masking
    num_triplets_real = torch.bincount(idx_e1, minlength=row.size(0))

    return (
        torch.stack([idx_i, idx_j, idx_k]),
        num_triplets_real.to(torch.long),
        edx_1st,
        edx_2nd,
    )
