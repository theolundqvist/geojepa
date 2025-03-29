import argparse
import pandas as pd
import pickle
import os


def run(embeddings_file, output_file, tag_counts_file, csv_file=None, index_offset=0):
    """
    Filters embeddings based on tag counts and saves the filtered embeddings.

    Parameters:
    - embeddings_file (str): Path to the pickle file containing base embeddings.
    - output_file (str): Path to save the filtered embeddings as a pickle file.
    - tag_counts_file (str): Path to the file containing tag counts.
    - csv_file (str, optional): Path to save the filtered embeddings as a CSV file.
    - index_offset (int, optional): Number of positions to adjust tag_counts indices when mapping to embeddings.
    """
    # Check if embeddings_file exists
    if not os.path.isfile(embeddings_file):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

    # Load embeddings
    try:
        with open(embeddings_file, "rb") as f:
            embeddings_df = pickle.load(f)
        print(f"Loaded embeddings from {embeddings_file}. Shape: {embeddings_df.shape}")
    except Exception as e:
        raise IOError(f"Error loading embeddings file: {e}")

    # Check if tag_counts_file exists
    if not os.path.isfile(tag_counts_file):
        raise FileNotFoundError(f"Tag counts file not found: {tag_counts_file}")

    # Load tag counts
    try:
        # Attempt to load as pickle first
        with open(tag_counts_file, "rb") as f:
            tag_counts = pickle.load(f)
        print(f"Loaded tag counts from {tag_counts_file} (pickle).")
    except Exception:
        try:
            # If not pickle, try reading as CSV
            tag_counts_df = pd.read_csv(tag_counts_file, index_col=0)
            if tag_counts_df.shape[1] == 1:
                # If there's only one column, convert to Series
                tag_counts = tag_counts_df.iloc[:, 0]
            else:
                # If multiple columns, assume the second column contains counts
                tag_counts = tag_counts_df.iloc[:, 0]
            print(f"Loaded tag counts from {tag_counts_file} (CSV).")
        except Exception as e:
            raise IOError(f"Error loading tag counts file: {e}")

    # Ensure tag_counts is a Series or dictionary
    if isinstance(tag_counts, pd.Series):
        tag_counts_dict = tag_counts.to_dict()
    elif isinstance(tag_counts, dict):
        tag_counts_dict = tag_counts
    else:
        raise TypeError(
            "tag_counts_file must be a pickle or CSV file containing a dictionary or pandas Series."
        )

    # Apply index offset
    if index_offset != 0:
        print(f"Applying index offset of {index_offset}.")
        adjusted_tag_counts_dict = {}
        for idx, count in tag_counts_dict.items():
            adjusted_idx = idx + index_offset
            if adjusted_idx in embeddings_df.index:
                adjusted_tag_counts_dict[adjusted_idx] = count
            else:
                print(
                    f"Warning: Adjusted index {adjusted_idx} not found in embeddings. Skipping."
                )
        tag_counts_dict = adjusted_tag_counts_dict
    else:
        print("No index offset applied.")

    # Filter embeddings where tag_count > 0
    print("Filtering embeddings based on tag counts...")
    filtered_indices = [idx for idx, count in tag_counts_dict.items() if count > 0]
    filtered_embeddings_df = embeddings_df.loc[filtered_indices]
    print(f"Filtered embeddings. New shape: {filtered_embeddings_df.shape}")

    # Save filtered embeddings to output_file
    try:
        with open(output_file, "wb") as f:
            pickle.dump(filtered_embeddings_df, f)
        print(f"Saved filtered embeddings to {output_file}.")
    except Exception as e:
        raise IOError(f"Error saving filtered embeddings to pickle file: {e}")

    # If csv_file is provided, save as CSV
    if csv_file:
        try:
            filtered_embeddings_df.to_csv(csv_file)
            print(f"Saved filtered embeddings to CSV file {csv_file}.")
        except Exception as e:
            raise IOError(f"Error saving filtered embeddings to CSV file: {e}")


def main():
    parser = argparse.ArgumentParser(
        prog="filter_embeddings",
        description="Filter embeddings file into a new file according to config file",
    )

    parser.add_argument(
        "-i",
        "--embeddings_file",
        type=str,
        help="File containing base embeddings",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="File to write new embeddings in",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--tag_counts_file",
        type=str,
        help="File containing all tag_counts, basis for pruning",
        required=True,
    )
    parser.add_argument(
        "-csv",
        "--csv_file",
        type=str,
        help="CSV file to write new embeddings in",
        required=False,
    )
    parser.add_argument(
        "-offset",
        "--index_offset",
        type=int,
        help="Index offset between tag_counts and embeddings",
        default=-2,
    )

    args = parser.parse_args()

    run(
        args.embeddings_file,
        args.output_file,
        args.tag_counts_file,
        args.csv_file,
        args.index_offset,
    )


if __name__ == "__main__":
    main()
