import argparse
import glob
from pathlib import Path
from typing import Tuple

import pandas as pd
import sys
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import src.data.components.processed_tile_group_pb2 as pbf
from scipy import stats



def get_parent_tile(new_x, new_y) -> Tuple[int, int]:
    zoom_diff = 16 - 14
    old_x = new_x // (2**zoom_diff)
    old_y = new_y // (2**zoom_diff)
    return old_x, old_y


def main():
    parser = argparse.ArgumentParser(
        description="Summary statistics for label and feature count files."
    )
    parser.add_argument("task", type=str, help="Path to the task dir")
    args = parser.parse_args()
    dir = Path(args.task)
    label_file = dir / "labels.txt"
    pbf_dir = dir

    # Step 1: Read the Labels File
    try:
        # Read the file into a pandas DataFrame
        data = pd.read_csv(label_file, sep=":", header=None, names=["tile_id", "label"])
    except FileNotFoundError:
        print(f"Error: File '{label_file}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{label_file}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{label_file}': {e}")
        sys.exit(1)

    # Ensure label column is numeric
    try:
        data["label"] = pd.to_numeric(data["label"], errors="coerce")
    except Exception as e:
        print(f"Error converting labels to numeric: {e}")
        sys.exit(1)

    # Drop any rows with NaN labels
    num_missing = data["label"].isna().sum()
    if num_missing > 0:
        print(
            f"Warning: {num_missing} labels could not be converted to numbers and will be ignored."
        )
        data = data.dropna(subset=["label"])

    # Step 2: Read .pbf Files and Extract Feature Counts
    feature_count = {}
    print("\nProcessing .pbf files to extract feature counts...")
    pbf_files = glob.glob(f"{pbf_dir}/**/*.pbf", recursive=True)
    if not pbf_files:
        print(f"Warning: No .pbf files found in directory '{pbf_dir}'.")
    for file in tqdm(pbf_files, f"Processing files in {pbf_dir}"):
        tg = pbf.TileGroup()
        try:
            with open(file, "rb") as f:
                tg.ParseFromString(f.read())
        except Exception as e:
            print(f"Warning: Failed to parse '{file}': {e}")
            continue
        for tile in tg.tiles:
            tile_id = f"{tile.zoom}_{tile.x}_{tile.y}"
            feature_count[tile_id] = len(tile.features)
    print(f"Processed {len(feature_count)} tiles from .pbf files.")

    # Step 3: Merge Feature Counts with Labels
    print("\nMerging feature counts with labels...")
    data["feature_count"] = data["tile_id"].map(feature_count)
    missing_features = data["feature_count"].isna().sum()
    if missing_features > 0:
        print(
            f"Warning: {missing_features} tiles in labels file do not have corresponding feature counts."
        )
        data = data.dropna(subset=["feature_count"])
    data["feature_count"] = data["feature_count"].astype(int)

    # Step 4: Compute Label Statistics (Existing)
    labels = data["label"]
    total_tiles = len(labels)
    max_label = labels.max()
    min_label = labels.min()
    mean_label = labels.mean()
    std_label = labels.std()
    num_zeros = (labels == 0).sum()
    num_unique = labels.nunique()

    # Additional Label Statistics
    median_label = labels.median()
    label_counts = labels.value_counts().sort_index()

    # Step 5: Compute Feature Count Statistics
    features = data["feature_count"]
    total_features = features.sum()
    max_features = features.max()
    min_features = features.min()
    mean_features = features.mean()
    std_features = features.std()
    median_features = features.median()
    unique_features = features.nunique()

    # Correlation between labels and feature counts
    correlation, p_value = stats.pearsonr(data["label"], data["feature_count"])

    # Feature Distribution
    feature_counts = features.value_counts().sort_index()

    # Per-Label Feature Count Statistics
    per_label_features = data.groupby("label")["feature_count"]
    mean_features_per_label = per_label_features.mean()
    std_features_per_label = per_label_features.std()
    median_features_per_label = per_label_features.median()

    # Step 6: Print Summary
    print("\n=== Label Summary ===")
    print(f"Total number of tiles: {total_tiles}")
    print(f"Maximum label: {max_label}")
    print(f"Minimum label: {min_label}")
    print(f"Mean label: {mean_label:.2f}")
    print(f"Standard Deviation: {std_label:.2f}")
    print(f"Median label: {median_label}")
    print(f"Number of zeros: {num_zeros}")
    print(f"Number of unique labels: {num_unique}")

    # Calculate MAE for predicting the most common label (Mode)
    mode_label = labels.mode()[0]
    mae_mode = (labels - mode_label).abs().mean()
    # Calculate MAE for predicting the mean label
    mae_mean = (labels - mean_label).abs().mean()
    # Calculate MAE for predicting the median label (Best Guess)
    mae_median = (labels - median_label).abs().mean()

    print(num_zeros / total_tiles)
    if num_zeros / total_tiles > 0.9999:
        loss = torch.nn.L1Loss()
    else:
        loss = 0.0
    weighted_loss = loss(torch.zeros(len(labels)), torch.tensor(labels))

    # Print MAE results
    print("\n=== Mean Absolute Error (MAE) ===")
    print(
        f"MAE if predicting the most common label (Mode = {mode_label}): {mae_mode:.2f}"
    )
    print(f"MAE if predicting the mean label (Mean = {mean_label:.2f}): {mae_mean:.2f}")
    print(
        f"MAE if predicting the median label (Median = {median_label}): {mae_median:.2f}"
    )
    print(
        f"WeightedMSE if predicting the most common label (Mode = {mode_label}): {weighted_loss:.2f}"
    )
    print(
        f"WeightedMSE if predicting the mean label (Mean = {mean_label}): {loss(torch.ones(len(labels)) * mean_label, torch.tensor(labels)):.2f}"
    )
    print(
        f"WeightedMSE if predicting 1.0: {loss(torch.ones(len(labels)), torch.tensor(labels)):.2f}"
    )
    print(
        f"WeightedMSE if predicting 2.0: {loss(torch.ones(len(labels)) * 2, torch.tensor(labels)):.2f}"
    )
    print(
        f"WeightedMSE if predicting 2.5: {loss(torch.ones(len(labels)) * 2.5, torch.tensor(labels)):.2f}"
    )
    print(
        f"WeightedMSE if predicting 3.0: {loss(torch.ones(len(labels)) * 3, torch.tensor(labels)):.2f}"
    )
    print(
        f"WeightedMSE if predicting 4.0: {loss(torch.ones(len(labels)) * 4, torch.tensor(labels)):.2f}"
    )

    print("\n=== Label Distribution ===")
    for label, count in label_counts.items():
        percentage = (count / total_tiles) * 100
        print(f"Label {label}: {count} ({percentage:.2f}%)")

    print("\n=== Feature Count Summary ===")
    print(f"Total number of features: {total_features}")
    print(f"Maximum feature count per tile: {max_features}")
    print(f"Minimum feature count per tile: {min_features}")
    print(f"Mean feature count per tile: {mean_features:.2f}")
    print(f"Standard Deviation of feature counts: {std_features:.2f}")
    print(f"Median feature count per tile: {median_features}")
    print(f"Number of unique feature counts: {unique_features}")

    print("\n=== Correlation ===")
    print(
        f"Pearson correlation between labels and feature counts: {correlation:.4f} (p-value: {p_value:.4e})"
    )

    print("\n=== Feature Distribution ===")
    for fc, count in list(feature_counts.items())[:20]:
        percentage = (count / total_tiles) * 100
        print(f"Feature Count {fc}: {count} ({percentage:.2f}%)")
    print("-----------------")
    count_to_feature = {count: f for f, count in feature_count.items()}
    for fc, count in list(feature_counts.items())[-20:]:
        percentage = (count / total_tiles) * 100
        tile = count_to_feature[count]
        parent = "14_" + "_".join(
            list(
                map(
                    str,
                    get_parent_tile(int(tile.split("_")[1]), int(tile.split("_")[2])),
                )
            )
        )
        print(f"Feature Count {fc}: {count} ({percentage:.2f}%), {tile} -> {parent}")
    # print("\n... some of the worst feature counts ...")

    # for k, v in list(feature_counts.items())[0:10]:
    #     print(k, v)
    # for k, v in list(feature_count.items())[-10:]:
    #     print(k, v)
    #     print("14_" + "_".join(list(map(str, get_parent_tile(int(k.split("_")[1]), int(k.split("_")[2]))))))

    print("\n=== Per-Label Feature Count Statistics ===")
    print(f"{'Label':<10}{'Mean':<10}{'Std':<10}{'Median':<10}{'Mode'}")
    for label in sorted(data["label"].unique()):
        mean_fc = mean_features_per_label[label]
        std_fc = std_features_per_label[label]
        median_fc = median_features_per_label[label]
        print(f"{label:<10}{mean_fc:<10.2f}{std_fc:<10.2f}{median_fc:<10}")

    # Step 7: Plotting
    print("\nGenerating plots...")

    # Plot 1: Label Distribution
    plt.figure(figsize=(10, 6))
    label_counts.plot(kind="bar", color="skyblue")
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Plot 2: Feature Count Distribution
    plt.figure(figsize=(10, 6))
    features.plot(kind="hist", bins=30, color="salmon", edgecolor="black", alpha=0.7)
    plt.title("Feature Count Distribution")
    plt.xlabel("Number of Features per Tile")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Plot 3: Feature Count vs Label (Scatter Plot)
    plt.figure(figsize=(10, 6))
    plt.scatter(
        data["label"], data["feature_count"], alpha=0.6, edgecolors="w", linewidth=0.5
    )
    plt.title("Feature Count vs Label")
    plt.xlabel("Label")
    plt.ylabel("Number of Features per Tile")
    plt.tight_layout()
    plt.show()

    # Plot 4: Box Plot of Feature Counts per Label
    plt.figure(figsize=(12, 8))
    data.boxplot(column="feature_count", by="label", grid=False)
    plt.title("Feature Count Distribution per Label")
    plt.suptitle("")  # Suppress the automatic title to avoid duplication
    plt.xlabel("Label")
    plt.ylabel("Number of Features per Tile")
    plt.tight_layout()
    plt.show()

    print("Summary and plots generated successfully.")


if __name__ == "__main__":
    main()
