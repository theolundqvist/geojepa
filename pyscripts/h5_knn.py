import h5py
import rootutils
from pacmap import PaCMAP
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from src.data.components.raw_tile_dataset import load_images_from_path


def get_parent_tile(x: int, y: int, current_z: int = 16, parent_z: int = 14):
    """
    Calculate the parent tile coordinates for a given tile.

    Parameters:
    - x (int): X coordinate of the current tile.
    - y (int): Y coordinate of the current tile.
    - current_z (int): Current zoom level (default: 16).
    - parent_z (int): Parent zoom level (default: 14).

    Returns:
    - Tuple[int, int]: Parent tile's (x, y) coordinates.
    """
    assert current_z > parent_z, "Current zoom must be greater than parent zoom."
    zoom_diff = current_z - parent_z
    parent_x = x // (2**zoom_diff)
    parent_y = y // (2**zoom_diff)
    return parent_x, parent_y


def load_embeddings(cls_path, feat_path=None):
    """
    Load embeddings from HDF5 files.

    Parameters:
    - cls_path (str): Path to cls.h5
    - feat_path (str, optional): Path to feat.h5

    Returns:
    - embeddings_dict (dict): Mapping from group names to embeddings
    - embedding_dim (int): Dimension of the embeddings
    """
    embeddings_dict = {}
    use_features = False

    id_path = cls_path.replace("cls.h5", "identifiers.h5")
    with h5py.File(id_path, "r") as f:
        ids = f["identifiers"][:]

    if isinstance(ids[0], bytes):
        ids = [id.decode("utf-8") for id in ids]
    else:
        ids = [str(id) for id in ids]

    id_map = {tile_name: idx for idx, tile_name in enumerate(ids)}
    del ids

    if feat_path and os.path.exists(feat_path):
        with h5py.File(feat_path, "r") as feat_h5:
            print(f"Loading features from {feat_path}")
            for group_name in feat_h5:
                embeddings_dict[group_name] = feat_h5[group_name]["features"][:]
                print(
                    f"  features data shape: {embeddings_dict[group_name].shape}: {group_name}"
                )
                print(
                    f"  features data shape: {embeddings_dict[group_name][1].shape}: {group_name}"
                )
        use_features = True
    else:
        with h5py.File(cls_path, "r") as cls_h5:
            print(f"Loading cls from {cls_path}")
            for tile_name, id in id_map.items():
                embeddings_dict[tile_name] = cls_h5["cls"][id][:]

    # Determine embedding dimension
    if embeddings_dict:
        first_key = next(iter(embeddings_dict))
        embedding_dim = embeddings_dict[first_key].shape[-1]
        print(f"Embedding dimension: {embedding_dim}")
    else:
        raise ValueError("No embeddings found in the provided HDF5 files.")

    return embeddings_dict, embedding_dim


def select_group(embeddings_dict, group_name):
    """
    Select a specific group from the embeddings.

    Parameters:
    - embeddings_dict (dict): Mapping from group names to embeddings
    - group_name (str): The group name to select

    Returns:
    - selected_embedding (np.ndarray): The embedding of the selected group
    """
    if group_name not in embeddings_dict:
        raise ValueError(f"Group name '{group_name}' not found in embeddings.")
    return embeddings_dict[group_name]


def perform_knn_search(embeddings, query_embedding, k=5):
    """
    Perform kNN search to find the nearest neighbors.

    Parameters:
    - embeddings (np.ndarray): All embeddings (shape: num_samples x embedding_dim)
    - query_embedding (np.ndarray): The embedding to query (shape: embedding_dim,)
    - k (int): Number of nearest neighbors to find

    Returns:
    - distances (np.ndarray): Distances to the nearest neighbors
    - indices (np.ndarray): Indices of the nearest neighbors
    """
    # Initialize NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(embeddings)
    distances, indices = nbrs.kneighbors([query_embedding])

    # Exclude the first neighbor since it is the query itself
    return distances[0][1:], indices[0][1:]


image_dict = {}


def get_image(tile_name, image_dir, default_image_path, zoom=0.25):
    """
    Load the image corresponding to a group name.

    Parameters:
    - group_name (str): The group name.
    - image_dir (str): Directory where images are stored.
    - default_image_path (str): Path to a default image if group image is missing.
    - zoom (float): Scaling factor for the image.

    Returns:
    - image (OffsetImage): The image to embed in the plot.
    """
    z, x, y = tuple(map(int, tile_name.split("_")))
    if (x, y) not in image_dict:
        group_coord = get_parent_tile(x, y)
        image_path = os.path.join(
            image_dir, f"14_{group_coord[0]}_{group_coord[1]}.webp"
        )
        temp_dict = load_images_from_path(image_path)
        image_dict.update(temp_dict)
    image = image_dict[x, y]
    np_img = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(np_img)
    img = img.convert("RGBA")  # Ensure image has alpha channel
    return OffsetImage(img, zoom=zoom)


def visualize_all_groups_with_images(
    embeddings,
    group_names,
    image_dir,
    default_image_path,
    query_index,
    neighbor_indices,
    zoom=0.25,
):
    """
    Visualize all groups on a 2D plane using UMAP, highlighting the query and its nearest neighbors with images.

    Parameters:
    - embeddings (np.ndarray): All embeddings
    - group_names (list): List of all group names
    - image_dir (str): Directory containing group images
    - default_image_path (str): Path to default image for missing groups
    - query_index (int): Index of the query embedding
    - neighbor_indices (list): Indices of nearest neighbors
    - zoom (float): Zoom level for all images
    """
    # Reduce dimensions to 2D using UMAP instead of PCA
    # reducer = umap.UMAP(n_components=2, random_state=42)
    # reduced = reducer.fit_transform(embeddings)
    pacmap_model = PaCMAP(n_components=2)
    reduced = pacmap_model.fit_transform(embeddings)
    print("UMAP dimensionality reduction completed.")

    # Create plot
    fig, ax = plt.subplots(figsize=(20, 16))
    ax.set_title(f'kNN Visualization for "{group_names[query_index]}"', fontsize=20)
    ax.set_xlabel("UMAP Component 1", fontsize=16)
    ax.set_ylabel("UMAP Component 2", fontsize=16)
    ax.grid(True)

    # Plot all groups as light gray points
    ax.scatter(
        reduced[:, 0], reduced[:, 1], color="black", alpha=1.0, label="All Groups"
    )

    # Embed images for all groups
    for idx, group_name in tqdm(
        enumerate(group_names), desc="Loading images", total=len(group_names)
    ):
        img = get_image(group_name, image_dir, default_image_path, zoom=zoom)
        if img:
            ab = AnnotationBbox(
                img,
                (reduced[idx, 0], reduced[idx, 1]),
                frameon=True,
                pad=0.3,
                bboxprops=dict(edgecolor="none"),  # Default no border
            )
            ax.add_artist(ab)

    # Highlight the query group
    query_coord = reduced[query_index]
    print(group_names[query_index])
    img = get_image(group_names[query_index], image_dir, default_image_path, zoom=zoom)
    print(img)
    ab_query = AnnotationBbox(
        img,
        (query_coord[0], query_coord[1]),
        frameon=True,
        bboxprops=dict(edgecolor="red", linewidth=2),
    )
    ax.add_artist(ab_query)
    ax.annotate(
        group_names[query_index],
        (query_coord[0], query_coord[1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        color="red",
        fontsize=12,
        fontweight="bold",
    )

    # Highlight nearest neighbors
    for idx in neighbor_indices:
        neighbor_coord = reduced[idx]
        ab_neighbor = AnnotationBbox(
            get_image(group_names[idx], image_dir, default_image_path, zoom=zoom),
            (neighbor_coord[0], neighbor_coord[1]),
            frameon=True,
            bboxprops=dict(edgecolor="blue", linewidth=2),
        )
        ax.add_artist(ab_neighbor)
        ax.annotate(
            group_names[idx],
            (neighbor_coord[0], neighbor_coord[1]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            color="blue",
            fontsize=10,
        )

    # Create a legend manually since we used images
    import matplotlib.patches as mpatches

    red_patch = mpatches.Patch(color="red", label="Query Group")
    blue_patch = mpatches.Patch(color="blue", label="kNN")
    gray_patch = mpatches.Patch(color="lightgray", label="All Groups")
    ax.legend(handles=[red_patch, blue_patch, gray_patch], loc="upper right")

    plt.show()


def main():
    # === Argument Parsing ===
    parser = argparse.ArgumentParser(
        description="Perform kNN search on HDF5 embeddings with image visualization using UMAP."
    )
    parser.add_argument(
        "-cls", "--cls_file", type=str, required=True, help="Path to cls.h5 file"
    )
    parser.add_argument(
        "-feat", "--feat_file", type=str, help="Path to feat.h5 file (optional)"
    )
    parser.add_argument(
        "-pbf",
        "--pbf_dir",
        type=str,
        help="Path to pbf dir (required if feat provided and visualize enabled)",
    )
    parser.add_argument(
        "-g", "--group", type=str, required=True, help="Group name to query"
    )
    parser.add_argument(
        "-k",
        "--neighbors",
        type=int,
        default=5,
        help="Number of nearest neighbors to find",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Visualize the nearest neighbors with images",
    )
    parser.add_argument(
        "-img",
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing group images (.webp files)",
    )
    parser.add_argument(
        "-def",
        "--default_image",
        type=str,
        default=None,
        help="Path to default image for missing groups",
    )
    parser.add_argument(
        "-z",
        "--zoom",
        type=float,
        default=0.25,
        help="Zoom level for all images in the plot (default: 0.25)",
    )  # Added zoom argument
    args = parser.parse_args()

    cls_h5_path = args.cls_file
    feat_h5_path = args.feat_file
    pbf_path = args.pbf_dir
    group_to_query = args.group
    k = args.neighbors
    visualize = args.visualize
    image_dir = args.image_dir
    default_image_path = args.default_image
    zoom = args.zoom  # Retrieve zoom level

    # === Validate Image Arguments if Visualization is Enabled ===
    if visualize:
        if not image_dir:
            print(
                "Error: Image directory must be provided with --image_dir when using --visualize."
            )
            return
        if feat_h5_path and not pbf_path:
            print(
                "Error: PBF directory must be provided with --pbf_dir when using --visualize and --feat_file."
            )
            return
        if not os.path.isdir(image_dir):
            print(
                f"Error: The specified image directory '{image_dir}' does not exist or is not a directory."
            )
            return
        if default_image_path and not os.path.exists(default_image_path):
            print(
                f"Warning: The specified default image '{default_image_path}' does not exist. Missing group images will not be displayed."
            )
            default_image_path = None

    # === Load Embeddings ===
    try:
        embeddings_dict, embedding_dim = load_embeddings(cls_h5_path, feat_h5_path)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return

    # === Prepare Data for kNN ===
    group_names = list(embeddings_dict.keys())
    all_embeddings = np.stack([embeddings_dict[name] for name in group_names])

    # === Check if the group exists ===
    if group_to_query not in embeddings_dict:
        print(f"Group '{group_to_query}' not found in the embeddings.")
        print("Available groups:")
        for name in group_names:
            print(f" - {name}")
        return

    # === Select Query Group ===
    selected_embedding = embeddings_dict[group_to_query]
    print(f"Selected group: {group_to_query}")
    print(f"Embedding shape: {selected_embedding.shape}")

    # === Perform kNN Search ===
    distances, indices = perform_knn_search(all_embeddings, selected_embedding, k=k)
    nearest_groups = [group_names[idx] for idx in indices]

    # === Display Results ===
    print(f"\nTop {k} nearest neighbors for '{group_to_query}':")
    for i, (dist, name) in enumerate(zip(distances, nearest_groups), 1):
        print(f"{i}. {name} (Distance: {dist:.4f})")

    # === Optional: Visualize Neighbors ===
    if visualize:
        query_index = group_names.index(group_to_query)
        neighbor_indices = indices
        visualize_all_groups_with_images(
            all_embeddings,
            group_names,
            image_dir,
            default_image_path,
            query_index,
            neighbor_indices,
            zoom=zoom,  # Pass zoom level to the visualization function
        )


if __name__ == "__main__":
    main()
