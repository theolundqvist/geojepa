import numpy as np
import pandas as pd
import argparse
import sys
import os


def calculate_mse_mae(df):
    # Filter out rows where 'expected' is 'N/A'
    print(df)
    filtered_df = df[df["expected"].apply(lambda x: not (pd.isna(x)))]

    # Convert columns to numeric types
    y_true = np.array(filtered_df["expected"])
    y_pred = np.array(filtered_df["predicted"])

    # # Calculate MSE and MAE
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))

    return mse, mae


def load_labels(file_path):
    labels = {}
    with open(file_path, "r") as file:
        for line in file:
            # Split by ':' to separate the image name and the number of signals
            name, signals = line.strip().split(":")
            labels[name] = int(float(signals))
    return labels


def main(directory):
    if not os.path.isdir(directory):
        print(f"Directory {directory} does not exist.")
        sys.exit(1)

    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):  # Process only .pkl files
            print(f"\n{filename}")
            # Load the DataFrame from the pickle file
            p = os.path.join(directory, filename)
            df_loaded = pd.read_pickle(p)
            print(df_loaded)
            df_loaded.to_csv(f"{p}.csv")

            # Calculate MSE and MAE
            mse, mae = calculate_mse_mae(df_loaded[1:])

            # Display the results using the filename as the label
            print(f"mse: {mse}, mae: {mae}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="visualizer", description="Print GPT results")
    parser.add_argument(
        "input_dir", type=str, help="Path to the directory containing PBF files"
    )
    args = parser.parse_args()
    main(args.input_dir)
