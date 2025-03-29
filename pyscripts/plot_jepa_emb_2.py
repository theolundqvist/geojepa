import argparse
import rootutils

# Initialize the project root (ensure this is correctly set up in your environment)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import matplotlib.pyplot as plt
from pacmap import PaCMAP
import logging

from h5_read import read_all_h5


def visualize_representation_space(embeddings):
    try:
        pacmap_model = PaCMAP(n_components=2)
        e = pacmap_model.fit_transform(embeddings)
        plt.figure(figsize=(3, 3))
        plt.scatter(e[:, 0], e[:, 1], s=30, cmap="black", alpha=0.6)
        plt.xlabel("")
        plt.ylabel("")
        plt.axis("off")
        plt.title("Embedding PaCMAP")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"PaCMAP visualization failed: {e}")


def main(args):
    features = read_all_h5(args.i)  # Assume this returns a list of embedding matrices
    for embeddings in features:
        print(embeddings.shape)
        embeddings = embeddings.squeeze()
        variance = embeddings.var(dim=-1)
        print(variance.shape)
        indices = variance.argsort(dim=0)
        print(indices.shape)
        embeddings = embeddings[indices]
        plt.imshow(embeddings, cmap="viridis", interpolation="nearest")
        plt.colorbar()  # Adds a colorbar to the side
        plt.title("Matrix Heatmap using imshow")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.show()
        visualize_representation_space(embeddings)


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
