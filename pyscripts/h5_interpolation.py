import h5py
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt


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

    if feat_path and os.path.exists(feat_path):
        with h5py.File(feat_path, "r") as feat_h5:
            print(f"Loading features from {feat_path}")
            for group_name in feat_h5:
                embeddings_dict[group_name] = feat_h5[group_name]["features"][:]
    else:
        with h5py.File(cls_path, "r") as cls_h5:
            print(f"Loading cls from {cls_path}")
            for group_name in cls_h5:
                embeddings_dict[group_name] = cls_h5[group_name]["cls"][:]

    # Determine embedding dimension
    if embeddings_dict:
        first_key = next(iter(embeddings_dict))
        embedding_dim = embeddings_dict[first_key].shape[-1]
        print(f"Embedding dimension: {embedding_dim}")
    else:
        raise ValueError("No embeddings found in the provided HDF5 files.")

    return embeddings_dict, embedding_dim


def select_groups(embeddings_dict, group1, group2):
    """
    Select two specific groups from the embeddings.

    Parameters:
    - embeddings_dict (dict): Mapping from group names to embeddings
    - group1 (str): The first group name
    - group2 (str): The second group name

    Returns:
    - embedding1 (np.ndarray): Embedding of the first group
    - embedding2 (np.ndarray): Embedding of the second group
    """
    if group1 not in embeddings_dict:
        raise ValueError(f"Group name '{group1}' not found in embeddings.")
    if group2 not in embeddings_dict:
        raise ValueError(f"Group name '{group2}' not found in embeddings.")

    embedding1 = embeddings_dict[group1]
    embedding2 = embeddings_dict[group2]

    return embedding1, embedding2


def interpolate_embeddings(embedding1, embedding2, steps=10, method="linear"):
    """
    Perform interpolation between two embeddings.

    Parameters:
    - embedding1 (np.ndarray): Starting embedding
    - embedding2 (np.ndarray): Ending embedding
    - steps (int): Number of interpolation steps
    - method (str): Interpolation method ('linear' or 'slerp')

    Returns:
    - interpolated_embeddings (list of np.ndarray): List of interpolated embeddings
    """
    interpolated_embeddings = []
    for idx, alpha in enumerate(np.linspace(0, 1, steps)):
        if method == "linear":
            interp = (1 - alpha) * embedding1 + alpha * embedding2
        elif method == "slerp":
            interp = slerp(alpha, embedding1, embedding2)
        else:
            raise ValueError("Interpolation method must be 'linear' or 'slerp'")
        interpolated_embeddings.append(interp)
    return interpolated_embeddings


def slerp(val, low, high):
    """
    Perform spherical linear interpolation between two vectors.

    Parameters:
    - val (float): Interpolation factor between 0 and 1.
    - low (np.ndarray): Starting vector.
    - high (np.ndarray): Ending vector.

    Returns:
    - interpolated (np.ndarray): Interpolated vector.
    """
    omega = np.arccos(
        np.clip(
            np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1.0, 1.0
        )
    )
    if omega == 0:
        return low
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def save_interpolated_embeddings(
    interpolated_embeddings, output_path, dataset_name="interpolated_features"
):
    """
    Save interpolated embeddings to an HDF5 file.

    Parameters:
    - interpolated_embeddings (list of np.ndarray): List of interpolated embeddings
    - output_path (str): Path to the output HDF5 file
    - dataset_name (str): Name of the dataset to store embeddings
    """
    with h5py.File(output_path, "w") as h5file:
        for idx, embedding in enumerate(interpolated_embeddings):
            group_name = f"interp_{idx + 1}"
            grp = h5file.create_group(group_name)
            grp.create_dataset(dataset_name, data=embedding)
            # Print step and corresponding group
            print(f"Step{idx + 1}: {group_name}")
    print(f"Interpolated embeddings saved to {output_path}")


def visualize_interpolation(interpolated_embeddings, embedding1, embedding2):
    """
    Visualize the interpolation using PCA for dimensionality reduction.

    Parameters:
    - interpolated_embeddings (list of np.ndarray): List of interpolated embeddings
    - embedding1 (np.ndarray): Starting embedding
    - embedding2 (np.ndarray): Ending embedding
    """
    from sklearn.decomposition import PCA

    # Combine all embeddings for PCA
    combined = np.vstack([embedding1, interpolated_embeddings, embedding2])

    # Reduce dimensions to 2D
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(combined)

    # Plot
    plt.figure(figsize=(10, 8))
    # Plot the interpolation path
    plt.plot(reduced[0, 0], reduced[0, 1], "ro", label="Start")  # Starting point
    plt.plot(reduced[-1, 0], reduced[-1, 1], "go", label="End")  # Ending point
    plt.plot(
        reduced[1:-1, 0], reduced[1:-1, 1], "bo-", label="Interpolations"
    )  # Interpolated points
    plt.title("Interpolation Between Two Embeddings")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()


def list_available_groups(embeddings_dict):
    """
    Print all available group names.

    Parameters:
    - embeddings_dict (dict): Mapping from group names to embeddings
    """
    print("Available groups:")
    for name in embeddings_dict.keys():
        print(f" - {name}")


def main():
    # === Argument Parsing ===
    parser = argparse.ArgumentParser(
        description="Interpolate between two groups in HDF5 embeddings."
    )
    parser.add_argument(
        "-cls", "--cls_file", type=str, required=True, help="Path to cls.h5 file"
    )
    parser.add_argument(
        "-feat", "--feat_file", type=str, help="Path to feat.h5 file (optional)"
    )
    parser.add_argument(
        "-g1",
        "--group1",
        type=str,
        required=True,
        help="First group name for interpolation",
    )
    parser.add_argument(
        "-g2",
        "--group2",
        type=str,
        required=True,
        help="Second group name for interpolation",
    )
    parser.add_argument(
        "-s", "--steps", type=int, default=10, help="Number of interpolation steps"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="interpolated_embeddings.h5",
        help="Output HDF5 file for interpolated embeddings",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        choices=["linear", "slerp"],
        default="linear",
        help="Interpolation method: 'linear' or 'slerp'",
    )
    parser.add_argument(
        "-v", "--visualize", action="store_true", help="Visualize the interpolation"
    )
    args = parser.parse_args()

    cls_h5_path = args.cls_file
    feat_h5_path = args.feat_file
    group1 = args.group1
    group2 = args.group2
    steps = args.steps
    output_path = args.output
    method = args.method
    visualize = args.visualize

    # === Load Embeddings ===
    try:
        embeddings_dict, embedding_dim = load_embeddings(cls_h5_path, feat_h5_path)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        sys.exit(1)

    # === List Available Groups if Necessary ===
    if group1 not in embeddings_dict or group2 not in embeddings_dict:
        print("One or both specified groups not found.")
        list_available_groups(embeddings_dict)
        sys.exit(1)

    # === Select Groups ===
    try:
        embedding1, embedding2 = select_groups(embeddings_dict, group1, group2)
    except ValueError as ve:
        print(ve)
        list_available_groups(embeddings_dict)
        sys.exit(1)

    # === Perform Interpolation ===
    interpolated_embeddings = interpolate_embeddings(
        embedding1, embedding2, steps=steps, method=method
    )
    print(
        f"Performed {method} interpolation with {steps} steps between '{group1}' and '{group2}'."
    )

    # === Save Interpolated Embeddings ===
    save_interpolated_embeddings(interpolated_embeddings, output_path)

    # === Optional Visualization ===
    if visualize:
        visualize_interpolation(interpolated_embeddings, embedding1, embedding2)


if __name__ == "__main__":
    main()
