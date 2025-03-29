import argparse
import os
import csv
import pickle
from tqdm import tqdm


def csv_to_dict_flawed_data(file_path):
    """
    Converts a flawed CSV file into a dictionary. Handles rows where the `value` field
    contains multiple comma-separated values, ensuring no errors occur.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        dict: Dictionary created from the CSV file.
    """
    result_dict = {}
    with open(file_path, mode="r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in tqdm(reader, desc="Parsing row ", unit="row"):
            if len(row) < 3:
                raise ValueError(f"Row in {file_path} has fewer than 3 fields: {row}")

            # The first column is the key
            key = row[0]
            # All columns except the first and last are combined as the value field
            value = ",".join(row[1:-1])
            # The last column is the count
            try:
                count = int(row[-1])
            except ValueError:
                raise ValueError(
                    f"Count value in {file_path} is not an integer: {row[-1]}"
                )

            result_dict[(key, value)] = count
    return result_dict


def process_directory(input_dir):
    """
    Processes all flawed CSV files in the input directory, converts them to dictionaries,
    and saves them as .pkl files.

    Args:
        input_dir (str): Path to the input directory containing the CSV files.
    """
    if not os.path.isdir(input_dir):
        raise ValueError(f"The path '{input_dir}' is not a valid directory.")

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            csv_path = os.path.join(input_dir, file_name)
            dict_data = csv_to_dict_flawed_data(csv_path)
            print(len(dict_data))

            pkl_name = f"{os.path.splitext(file_name)[0]}.pkl"
            pkl_path = os.path.join(input_dir, pkl_name)

            with open(pkl_path, "wb") as pkl_file:
                pickle.dump(dict_data, pkl_file)

            print(f"Processed {file_name} and saved as {pkl_name}")


def search_in_pkl(directory, query):
    """
    Searches for an exact match in keys or values in the `.pkl` files of a directory.
    If no query is provided, returns the 15 most common rows (sorted by count) for each file.

    Args:
        directory (str): Path to the directory containing `.pkl` files.
        query (str, optional): The exact query string to search for.

    Returns:
        dict: A dictionary where keys are filenames, and values are lists of tuples
              (key, value, count) sorted by count in descending order.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"The path '{directory}' is not a valid directory.")

    results_by_file = {}

    for file_name in os.listdir(directory):
        if file_name.endswith(".pkl"):
            pkl_path = os.path.join(directory, file_name)

            with open(pkl_path, "rb") as pkl_file:
                data_dict = pickle.load(pkl_file)

            # Sort by count, highest first
            if query:
                # Search for exact matches in key or value
                matches = [
                    (key, value, count)
                    for (key, value), count in tqdm(
                        data_dict.items(), desc=f"Searching for {query}", unit="item"
                    )
                    if key.lower() == query or value.lower() == query
                ]
                matches = sorted(matches, key=lambda x: x[2], reverse=True)[:15]
            else:
                # Get the 15 most common rows sorted by count
                matches = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)[
                    :15
                ]
                # Convert back to (key, value, count) format
                matches = [(key, value, count) for (key, value), count in matches]

            if matches:
                results_by_file[file_name] = matches

    return results_by_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze generated cooccurrence files")
    parser.add_argument(
        "-i", "--input_dir", type=str, required=True, help="Input directory path"
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional input to compare with other file than cheat",
    )
    parser.add_argument(
        "--gen_pkl",
        action="store_true",
        help="Optional input, activate if pkl files haven't been generated",
    )
    args = parser.parse_args()
    if args.gen_pkl:
        process_directory(args.input_dir)
    else:
        matches = search_in_pkl(args.input_dir, args.query)
        # Print results grouped by file.
        for file_name, rows in matches.items():
            print(f"File: {file_name}")
            for key, value, count in rows:
                print(f"  Key: {key}, Value: {value}, Count: {count}")
