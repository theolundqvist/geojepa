import pickle
import pandas as pd
from pandas import DataFrame

# Define file paths
emb_path = "data/tiles/embeddings.pkl"
pruned_emb_path = "data/tiles/pruned_embeddings.pkl"
pruned_emb_csv = "data/tiles/pruned_embeddings.csv"

# Load the embeddings dictionary using a context manager
try:
    with open(emb_path, "rb") as f:
        embeddings: DataFrame = pickle.load(f)
    print("Loaded embeddings successfully.")
except FileNotFoundError:
    print(f"Error: The file {emb_path} was not found.")
    raise
except Exception as e:
    print(f"An error occurred while loading embeddings: {e}")
    raise

# Load the CSV file containing keys to retain
try:
    emb_csv = pd.read_csv(pruned_emb_csv)
    print("Loaded CSV file successfully.")
except FileNotFoundError:
    print(f"Error: The file {pruned_emb_csv} was not found.")
    raise
except pd.errors.EmptyDataError:
    print("Error: The CSV file is empty.")
    raise
except Exception as e:
    print(f"An error occurred while reading the CSV file: {e}")
    raise

# Inspect the DataFrame to determine the correct column name
print("CSV Columns:", emb_csv.columns.tolist())

# **Important:** Replace 'key_column_name' with the actual column name that contains the keys
key_column_name = "key"  # Example column name

if key_column_name not in emb_csv.columns:
    raise ValueError(
        f"The specified key column '{key_column_name}' does not exist in the CSV file."
    )

# Convert the relevant column to a set for efficient lookups
keys_to_keep = set(emb_csv[key_column_name].astype(str))  # Ensure keys are strings

# Initialize the pruned embeddings dictionary
pruned_embeddings = embeddings.copy()
pruned_embeddings = pruned_embeddings[0:0]
i = 0

# Iterate through the embeddings and retain only those keys present in the CSV
# embeddings.set_index('key', inplace=True)
pruned_idx = {}
for i, row in emb_csv.iterrows():
    key = row["Unnamed: 0"]
    print(row.name, key)
    pruned_idx[key] = i
exit(0)
pruned_embeddings = pd.DataFrame(pruned_embeddings)

print(embeddings)
print(pruned_embeddings)


exit(0)
print(f"Total number of pruned embeddings: {i}")

# Save the pruned embeddings dictionary to a new pickle file
try:
    with open(pruned_emb_path, "wb") as f:
        pickle.dump(pruned_embeddings, f)
    print(f"Pruned embeddings saved successfully to {pruned_emb_path}.")
except Exception as e:
    print(f"An error occurred while saving pruned embeddings: {e}")
    raise
