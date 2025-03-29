import argparse
from itertools import islice
import rootutils

# Initialize the project root (ensure this is correctly set up in your environment)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # For numerical computations
from pacmap import PaCMAP
from scipy.stats import gaussian_kde
from tqdm import tqdm
import logging

from h5_read import read_all_h5
from sklearn.decomposition import PCA


def get_dist_with_pca(data: np.ndarray, n_components: int = 10) -> np.ndarray:
    """
    Apply PCA to reduce dimensionality before computing Gaussian KDE.

    Parameters:
        data (np.ndarray): 1D array of data points.
        n_components (int): Number of principal components.

    Returns:
        np.ndarray: Normalized probability density values.
    """
    pca = PCA(n_components=n_components)
    data_reduced = pca.fit_transform(data.reshape(-1, 1)).flatten()
    kde = gaussian_kde(data_reduced)
    grid = np.linspace(data_reduced.min(), data_reduced.max(), 100)
    p = kde(grid)
    p /= p.sum()
    return p


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_dist(data: np.ndarray) -> np.ndarray:
    """
    Compute the normalized probability density function using Gaussian KDE.

    Parameters:
        data (np.ndarray): 1D array of data points.

    Returns:
        np.ndarray: Normalized probability density values or np.nan array if KDE fails.
    """
    try:
        kde = gaussian_kde(data)
        grid = np.linspace(data.min(), data.max(), 100)
        p = kde(grid)
        if np.any(np.isnan(p)):
            raise ValueError("KDE returned NaN values.")
        p /= p.sum()  # Normalize to ensure the sum is 1
        return p
    except Exception as e:
        logging.error(f"Gaussian KDE failed: {e}")
        return np.full(100, np.nan)  # Return an array of NaNs to indicate failure


def get_collapse_metrics(X: np.ndarray) -> dict:
    """
    Compute collapse metrics between two randomly selected samples from X.

    Parameters:
        X (np.ndarray): 2D array where each row is a sample.

    Returns:
        dict: Dictionary containing KL divergence and Euclidean distance.
    """
    try:
        idx_1, idx_2 = np.random.randint(0, X.shape[0], size=2)

        data_1 = X[idx_1].flatten()
        data_2 = X[idx_2].flatten()

        p1 = get_dist_with_pca(data_1)
        p2 = get_dist_with_pca(data_2)

        # Check if KDE failed
        if np.isnan(p1).any() or np.isnan(p2).any():
            raise ValueError("Gaussian KDE returned NaN values.")

        kl_divergence = np.sum(
            p1 * np.log(p1 / (p2 + 1e-10))
        )  # Added epsilon to prevent log(0)
        euclidean_distance = np.linalg.norm(data_1 - data_2)

        collapse_metrics = {
            "KL": kl_divergence,
            "euclidean": euclidean_distance,
        }

        return collapse_metrics
    except Exception as e:
        logging.error(f"Error in get_collapse_metrics: {e}")
        return {"KL": np.nan, "euclidean": np.nan}


def visualize_representation_space(embeddings):
    """
    Visualize the representation space using PaCMAP dimensionality reduction.

    Parameters:
        embeddings (np.ndarray): 2D array of embeddings with shape (num_samples, embedding_dim).
    """
    num_points = min(50000, embeddings.shape[0])
    if num_points < embeddings.shape[0]:
        sampled_embeddings = embeddings[
            np.random.choice(embeddings.shape[0], num_points, replace=False)
        ]
    else:
        sampled_embeddings = embeddings

    try:
        # Initialize PaCMAP
        pacmap_model = PaCMAP(n_components=2)

        # Fit and transform
        embeddings_2d = pacmap_model.fit_transform(sampled_embeddings)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1], s=1, cmap="viridis", alpha=0.6
        )
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title("Representation Space Visualization")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"PaCMAP visualization failed: {e}")


def compute_uniformity(X: np.ndarray) -> float:
    """
    Compute the uniformity metric based on the log average exponential of pairwise distances.

    Parameters:
        X (np.ndarray): 2D array where each row is a sample.

    Returns:
        float: Uniformity score.
    """
    try:
        # Normalize embeddings to unit vectors
        X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)

        # Compute pairwise squared Euclidean distances
        # To avoid computing all pairwise distances (which can be memory intensive),
        # sample a subset of pairs.
        n_samples = min(10000, X_normalized.shape[0] * (X_normalized.shape[0] - 1) // 2)
        distances = []

        # Randomly sample pairs
        for _ in range(n_samples):
            idx_1, idx_2 = np.random.randint(0, X_normalized.shape[0], size=2)
            if idx_1 == idx_2:
                continue
            vec1 = X_normalized[idx_1]
            vec2 = X_normalized[idx_2]
            dist_sq = np.sum((vec1 - vec2) ** 2)
            distances.append(dist_sq)

        distances = np.array(distances)
        uniformity = np.log(np.mean(np.exp(-2 * distances)))

        return uniformity
    except Exception as e:
        logging.error(f"Uniformity computation failed: {e}")
        return np.nan


def plot_metrics(metrics_df: pd.DataFrame):
    """
    Plot various metrics over batches.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame containing metrics.
    """
    try:
        plt.figure(figsize=(12, 8))
        plt.plot(
            metrics_df["Batch"],
            metrics_df["Intra-feature Variance"],
            label="Intra-feature Variance",
        )
        plt.plot(
            metrics_df["Batch"],
            metrics_df["Inter-feature Variance"],
            label="Inter-feature Variance",
        )
        plt.xlabel("Batch")
        plt.ylabel("Variance")
        plt.title("Embedding Variances Over Batches")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(
            metrics_df["Batch"],
            metrics_df["KL Divergence"],
            label="KL Divergence",
            color="orange",
        )
        plt.xlabel("Batch")
        plt.ylabel("KL Divergence")
        plt.title("KL Divergence Over Batches")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(
            metrics_df["Batch"],
            metrics_df["Euclidean Distance"],
            label="Euclidean Distance",
            color="green",
        )
        plt.xlabel("Batch")
        plt.ylabel("Euclidean Distance")
        plt.title("Euclidean Distance Over Batches")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(
            metrics_df["Batch"],
            metrics_df["Uniformity"],
            label="Uniformity Score",
            color="red",
        )
        plt.xlabel("Batch")
        plt.ylabel("Uniformity")
        plt.title("Uniformity Over Batches")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Plotting metrics failed: {e}")


def perform_pacmap(embeddings):
    """
    Perform dimensionality reduction using PaCMAP and visualize the results.

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    try:
        # Parameters for the second set of embeddings
        # Initialize and fit PaCMAP for dimensionality reduction to 2D
        pacmap_model = PaCMAP(n_components=2, verbose=1)
        embeddings_2d = pacmap_model.fit_transform(embeddings)

        # Plot the 2D embeddings
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            s=1,
            alpha=0.5,
            cmap="viridis",
            # c=np.random.rand(num_samples)  # Replace with meaningful labels if available
        )
        plt.title("PaCMAP Dimensionality Reduction")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        cbar = plt.colorbar(
            scatter, label="Random Colors"
        )  # Update label if coloring by meaningful data
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"PaCMAP processing failed: {e}")


def main(args):
    """
    Main function to generate embeddings, compute variances, create heatmaps,
    perform dimensionality reduction, and visualize the results.

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments.
    """

    # ------------------------------
    # Part 1: Generate and Analyze Embeddings
    # ------------------------------
    try:
        features = read_all_h5(
            args.i
        )  # Assume this returns a list of embedding matrices
    except Exception as e:
        logging.error(f"Failed to read embeddings: {e}")
        return

    # Initialize lists to store metrics across embeddings (batches)
    batches = []
    intra_variances = []
    inter_variances = []
    dkl_values = []
    norm_values = []
    uniformity_scores = []
    variance_across_features = []  # New list for variance across features
    variance_across_hidden_features = []  # New list for variance across hidden features

    # Use tqdm for progress tracking
    for batch_idx, embedding_entry in enumerate(
        tqdm(islice(features, 3), desc="Processing Embeddings"), start=1
    ):
        try:
            # embedding_entry shape: (1, N, 384)
            embeddings = embedding_entry.squeeze(0)  # Shape: (N, 384)

            if embeddings.ndim != 2 or embeddings.shape[1] != 384:
                logging.warning(
                    f"Batch {batch_idx}: Unexpected embedding shape {embeddings.shape}. Skipping."
                )
                # Append NaN for all metrics to maintain list lengths
                intra_variances.append(np.nan)
                inter_variances.append(np.nan)
                dkl_values.append(np.nan)
                norm_values.append(np.nan)
                uniformity_scores.append(np.nan)
                variance_across_features.append(np.nan)
                variance_across_hidden_features.append(np.nan)
                batches.append(batch_idx)
                continue

            num_available = embeddings.shape[0]
            num_samples = min(args.num_samples, num_available)
            embeddings = embeddings[:num_samples]  # Shape: (num_samples, 384)

            # Compute intra-feature variance
            intra_var_per_feature = np.var(
                embeddings, axis=0
            )  # Variance per hidden feature across tokens
            intra_variance = np.mean(
                intra_var_per_feature
            )  # Average intra-feature variance
            intra_variances.append(intra_variance)

            # Compute inter-feature variance
            inter_var_per_token = np.var(
                embeddings, axis=1
            )  # Variance per token across hidden features
            inter_variance = np.mean(
                inter_var_per_token
            )  # Average inter-feature variance
            inter_variances.append(inter_variance)

            # Compute collapse metrics (KL divergence and Euclidean distance)
            collapse_metrics = get_collapse_metrics(embeddings)
            dkl = collapse_metrics["KL"]
            euclidean = collapse_metrics["euclidean"]
            dkl_values.append(dkl)
            norm_values.append(euclidean)

            # Compute uniformity
            uniformity = compute_uniformity(embeddings)
            uniformity_scores.append(uniformity)

            # Compute variances for both dimensions
            var_across_feats = np.var(embeddings, axis=0)  # Variance per hidden feature
            var_across_hidden = np.var(embeddings, axis=1)  # Variance per feature/token
            variance_across_features.append(var_across_feats)
            variance_across_hidden_features.append(var_across_hidden)

            # Compute centered embeddings and variances for heatmap (hidden features)
            centered_embeddings = embeddings - np.mean(
                embeddings, axis=0, keepdims=True
            )
            heatmap_values = np.var(centered_embeddings, axis=0)  # Shape: (384,)
            heatmap_matrix = heatmap_values.reshape(1, -1)

            # Plot the heatmap for hidden features
            plt.figure(figsize=(15, 2))  # Adjusted height for better visualization
            plt.imshow(heatmap_matrix, aspect="auto", cmap="viridis")
            plt.colorbar(label="Variance")
            plt.xlabel("Hidden Features (Embedding Dimensions)")
            plt.title(
                f"Inter-feature Variance Across Hidden Dimensions (Batch {batch_idx})"
            )
            plt.yticks([])  # Hide y-axis ticks since it's a single row
            plt.tight_layout()
            plt.show()

            # Compute centered embeddings and variances for heatmap (features)
            centered_embeddings_features = embeddings - np.mean(
                embeddings, axis=1, keepdims=True
            )
            heatmap_values_features = np.var(
                centered_embeddings_features, axis=1
            )  # Shape: (N,)
            # Reshape to (features, 1) for visualization
            heatmap_matrix_features = heatmap_values_features.reshape(-1, 1)

            # Plot the heatmap for features
            plt.figure(figsize=(2, 15))  # Adjusted width for better visualization
            plt.imshow(heatmap_matrix_features, aspect="auto", cmap="plasma")
            plt.colorbar(label="Variance")
            plt.ylabel("Features/Tokens")
            plt.title(f"Intra-feature Variance Across Features (Batch {batch_idx})")
            plt.xticks([])  # Hide x-axis ticks since it's a single column
            plt.tight_layout()
            plt.show()

            # Additionally, visualize the representation space
            visualize_representation_space(embeddings)

            # Log metrics for the current batch
            logging.info(f"Batch {batch_idx}:")
            logging.info(f"  Intra-feature Variance: {intra_variance:.4f}")
            logging.info(f"  Inter-feature Variance: {inter_variance:.4f}")
            logging.info(f"  Average KL Divergence: {dkl:.4e}")
            logging.info(f"  Average Euclidean Distance: {euclidean:.4f}")
            logging.info(f"  Uniformity Score: {uniformity:.4f}\n")

            # Append batch number
            batches.append(batch_idx)

        except Exception as e:
            logging.error(f"Error processing batch {batch_idx}: {e}")
            # Append NaN for all metrics to maintain list lengths
            intra_variances.append(np.nan)
            inter_variances.append(np.nan)
            dkl_values.append(np.nan)
            norm_values.append(np.nan)
            uniformity_scores.append(np.nan)
            variance_across_features.append(np.nan)
            variance_across_hidden_features.append(np.nan)
            batches.append(batch_idx)
            continue

    if not batches:
        logging.error("No valid embeddings were processed. Exiting.")
        return

    try:
        # Create a DataFrame to store all metrics
        metrics_df = pd.DataFrame(
            {
                "Batch": batches,
                "Intra-feature Variance": intra_variances,
                "Inter-feature Variance": inter_variances,
                "KL Divergence": dkl_values,
                "Euclidean Distance": norm_values,
                "Uniformity": uniformity_scores,
                "Variance Across Features": variance_across_features,
                "Variance Across Hidden Features": variance_across_hidden_features,
            }
        )

        # Verify that all columns have the same length
        lengths = [
            len(v)
            for v in [
                batches,
                intra_variances,
                inter_variances,
                dkl_values,
                norm_values,
                uniformity_scores,
                variance_across_features,
                variance_across_hidden_features,
            ]
        ]
        if len(set(lengths)) != 1:
            raise ValueError(f"Metric lists have differing lengths: {lengths}")

        # Save metrics to a CSV file
        metrics_df.to_csv("embedding_metrics_over_batches.csv", index=False)
        logging.info("Saved metrics to 'embedding_metrics_over_batches.csv'.")

        # Plot metrics
        plot_metrics(metrics_df)

    except Exception as e:
        logging.error(f"Failed to create or save DataFrame: {e}")

    # ------------------------------
    # Part 2: Dimensionality Reduction with PaCMAP
    # ------------------------------
    perform_pacmap(batches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and analyze embeddings.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results",
        help="Output directory for the plots",
    )
    parser.add_argument(
        "-i",
        type=str,
        default="data/embeddings/geojepa_m3_ti/pretraining_medium",
        help="Root directory of the data",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples to use for the plots (per embedding)",
    )

    args = parser.parse_args()
    main(args)
