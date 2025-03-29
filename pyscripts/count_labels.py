import argparse
import glob
from typing import Tuple

import pandas as pd
import sys
import os
from tqdm import tqdm

import src.data.components.processed_tile_group_pb2 as pbf
from scipy import stats



def get_parent_tile(new_x, new_y) -> Tuple[int, int]:
    zoom_diff = 16 - 14
    old_x = new_x // (2**zoom_diff)
    old_y = new_y // (2**zoom_diff)
    return old_x, old_y


def main(input_dir):
    print(input_dir)

    for subdir in [
        os.path.join(input_dir, d)
        for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ]:
        # Step 1: Read the Labels File
        try:
            label_file = os.path.join(subdir, "labels.txt")
            # Read the file into a pandas DataFrame
            data = pd.read_csv(
                label_file, sep=":", header=None, names=["tile_id", "label"]
            )
        except FileNotFoundError:
            print(f"Error: File '{label_file}' not found.")
            sys.exit(1)
        except pd.errors.EmptyDataError:
            print(f"Error: File '{label_file}' is empty.")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file '{label_file}': {e}")
            sys.exit(1)

        with open(f"{subdir}/stats.txt", "w") as logfile:
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
            pbf_files = glob.glob(f"{subdir}/**/*.pbf", recursive=True)
            if not pbf_files:
                print(f"Warning: No .pbf files found in directory '{subdir}'.")
            for file in tqdm(pbf_files, f"Processing task: {subdir}"):
                tg = pbf.TileGroup()
                try:
                    with open(file, "rb") as f:
                        tg.ParseFromString(f.read())
                except Exception as e:
                    print(f"Warning: Failed to parse '{f}': {e}")
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
            logfile.write("\n=== Label Summary ===\n")
            logfile.write(f"Total number of tiles: {total_tiles}\n")
            logfile.write(f"Maximum label: {max_label}\n")
            logfile.write(f"Minimum label: {min_label}\n")
            logfile.write(f"Mean label: {mean_label:.2f}\n")
            logfile.write(f"Standard Deviation: {std_label:.2f}\n")
            logfile.write(f"Median label: {median_label}\n")
            logfile.write(f"Number of zeros: {num_zeros}\n")
            logfile.write(f"Number of unique labels: {num_unique}\n")

            # Calculate MAE for predicting the most common label (Mode)
            mode_label = labels.mode()[0]
            mae_mode = (labels - mode_label).abs().mean()
            # Calculate MAE for predicting the mean label
            mae_mean = (labels - mean_label).abs().mean()
            # Calculate MAE for predicting the median label (Best Guess)
            mae_median = (labels - median_label).abs().mean()

            prop = num_zeros / total_tiles
            # loss = WeightedMSELoss(num_zeros/total_tiles)
            # weighted_loss = loss(torch.zeros(len(labels)),torch.tensor(labels))

            # Print MAE results
            logfile.write("\n=== Mean Absolute Error (MAE) ===\n")
            logfile.write(
                f"MAE if predicting the most common label (Mode = {mode_label}): {mae_mode:.2f}\n"
            )
            logfile.write(
                f"MAE if predicting the mean label (Mean = {mean_label:.2f}): {mae_mean:.2f}\n"
            )
            logfile.write(
                f"MAE if predicting the median label (Median = {median_label}): {mae_median:.2f}\n"
            )
            # logfile.write(f"WeightedMSE if predicting the most common label (Mode = {mode_label}): {weighted_loss:.2f}\n")
            # logfile.write(f"WeightedMSE if predicting the mean label (Mean = {mean_label}): {loss(torch.ones(len(labels))*mean_label, torch.tensor(labels)):.2f}\n")
            # logfile.write(f"WeightedMSE if predicting 1.0: {loss(torch.ones(len(labels)), torch.tensor(labels)):.2f}\n")
            # logfile.write(f"WeightedMSE if predicting 2.0: {loss(torch.ones(len(labels))*2, torch.tensor(labels)):.2f}\n")
            # logfile.write(f"WeightedMSE if predicting 2.5: {loss(torch.ones(len(labels))*2.5, torch.tensor(labels)):.2f}\n")
            # logfile.write(f"WeightedMSE if predicting 3.0: {loss(torch.ones(len(labels))*3, torch.tensor(labels)):.2f}\n")
            # logfile.write(f"WeightedMSE if predicting 4.0: {loss(torch.ones(len(labels))*4, torch.tensor(labels)):.2f}\n")

            logfile.write("\n=== Label Distribution ===\n")
            for label, count in label_counts.items():
                percentage = (count / total_tiles) * 100
                logfile.write(f"Label {label}: {count} ({percentage:.2f}%)\n")

            logfile.write("\n=== Feature Count Summary ===\n")
            logfile.write(f"Total number of features: {total_features}\n")
            logfile.write(f"Maximum feature count per tile: {max_features}\n")
            logfile.write(f"Minimum feature count per tile: {min_features}\n")
            logfile.write(f"Mean feature count per tile: {mean_features:.2f}\n")
            logfile.write(f"Standard Deviation of feature counts: {std_features:.2f}\n")
            logfile.write(f"Median feature count per tile: {median_features}\n")
            logfile.write(f"Number of unique feature counts: {unique_features}\n")

            logfile.write("\n=== Correlation ===\n")
            logfile.write(
                f"Pearson correlation between labels and feature counts: {correlation:.4f} (p-value: {p_value:.4e})\n"
            )

            logfile.write("\n=== Feature Distribution ===\n")
            for fc, count in list(feature_counts.items())[:10]:
                percentage = (count / total_tiles) * 100
                logfile.write(f"Feature Count {fc}: {count} ({percentage:.2f}%)\n")
            logfile.write("\n... some of the worst feature counts ...\n")
            i = 0
            for k, v in feature_count.items():
                if i >= 10:
                    break
                if v == 1:
                    i += 1
                    logfile.write(k + "\n")
                    logfile.write(
                        f"{get_parent_tile(int(k.split('_')[1]), int(k.split('_')[2]))}\n"
                    )

            logfile.write("\n=== Per-Label Feature Count Statistics ===\n")
            logfile.write(
                f"{'Label':<10}{'Mean':<10}{'Std':<10}{'Median':<10}{'Mode'}\n"
            )
            for label in sorted(data["label"].unique()):
                mean_fc = mean_features_per_label[label]
                std_fc = std_features_per_label[label]
                median_fc = median_features_per_label[label]
                logfile.write(
                    f"{label:<10}{mean_fc:<10.2f}{std_fc:<10.2f}{median_fc:<10}\n"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="label analyzer", description="Count labels and save to file"
    )
    parser.add_argument(
        "-i", "--task_dir", type=str, help="Path to the directory containing PBF files"
    )
    args = parser.parse_args()
    main(args.task_dir)
