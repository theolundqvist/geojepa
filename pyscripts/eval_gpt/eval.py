import base64
import json
import argparse
import glob
import os
import pandas as pd
import numpy as np
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

# Initialize OpenAI client
client = OpenAI()

#
# export OPENAI_API_KEY="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"
#


def calculate_mse_mae(df):
    # Filter out rows where 'expected' is 'N/A'
    filtered_df = df[df["expected"].apply(lambda x: not (pd.isna(x)))]

    # Convert columns to numeric types
    y_true = np.array(filtered_df["expected"])
    y_pred = np.array(filtered_df["predicted"])

    # # Calculate MSE and MAE
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))

    return mse, mae


# # Function to encode an image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Function to load traffic signals data from the labels file
def load_labels(file_path):
    labels = {}
    with open(file_path, "r") as file:
        for line in file:
            # Split by ':' to separate the image name and the number of signals
            name, signals = line.strip().split(":")
            labels[name] = int(float(signals))
    return labels


class Response(BaseModel):
    description: str
    prediction: float


class ResponseList(BaseModel):
    responses: list[Response]


def make_query(model, task, query, test):
    image_dir = f"../../data/tiles/small/tasks/{task}/smaller_images"

    # List of image file paths
    if test:
        image_paths = [
            f"{image_dir}/16_10476_25338.webp",
            f"{image_dir}/16_11127_26118.webp",
            f"{image_dir}/16_10717_25119.webp",
            f"{image_dir}/16_10718_25116.webp",
        ]
        for path in image_paths:
            os.system(f"open {path}")
    else:
        image_paths = glob.glob(os.path.join(image_dir, "*.webp"))

    # Encode images and store in a list
    base64_images = [encode_image(path) for path in image_paths]
    batches = [base64_images[i : i + 10] for i in range(0, len(base64_images), 10)]
    # for path in image_paths[:10]:
    #     os.system(f"open {path}")

    # for path in image_paths:
    #     print(path)

    labels = load_labels(f"{image_dir}/../labels.txt")

    responseBucket = []

    for batch in tqdm(batches, "Querying gpt API in batches"):
        # Prepare messages with encoded images
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{query}",
                    }
                ]
                + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                    }
                    for img in batch
                ],
            }
        ]

        # Make the API call
        response = client.beta.chat.completions.parse(
            model=model,
            temperature=0,
            messages=messages,
            response_format=ResponseList,
            # max_tokens=300,
        )

        # Access the parsed response and display structured output
        res = response.choices[0].message.parsed.responses
        responseBucket.extend(res)

    # Initialize an empty DataFrame with the necessary columns
    rows = []

    # Loop through image paths and results, appending each to the DataFrame
    for path, result in zip(image_paths, responseBucket):
        # Extract the image name from the path to use as a key in the lookup dictionary
        image_name = path.split("/")[-1].replace(".webp", "")
        expected = labels.get(image_name, "N/A")  # Default to "N/A" if not found
        if expected == "N/A":
            raise ValueError(f"{image_name} value not found")

        # Print the information
        # print(f"Image: {image_name}")
        # print(f"Reasoning: {result.description}")
        # print(f"Predicted: {result.prediction}")
        # print(f"Expected: {expected}\n")

        # Append the data to the DataFrame
        rows.append(
            {
                "file_name": image_name,
                "predicted": result.prediction,
                "expected": expected,
                "Reasoning": result.description,
            }
        )

    mse, mae = calculate_mse_mae(pd.DataFrame(rows))
    save_results_json(f"{task}/results.json", model, task, mse, mae, query)

    # Save to .pkl file
    rows.insert(
        0, {"file_name": task, "predicted": model, "expected": "", "Reasoning": query}
    )

    df = pd.DataFrame(rows)
    df.to_pickle(f"{task}/{model}.pkl")


def save_results_json(output_path, model, dataset, mse, mae, query):
    metrics = {"test/mse": mse, "test/mae": mae}
    results = {
        "model": model,
        "dataset": dataset,
        "cheat": "false",
        "tiny": metrics,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "query": query,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


def main(directory):
    if not os.path.isdir(f"./{directory}"):
        os.makedirs(directory)
    task = directory

    if task == "traffic_signals":
        query = (
            "For each provided satellite image, estimate the number of traffic signal crossings. "
            "Consider area density, road types, and visible pedestrian activity. Analyze each image individually."
            "Images for analysis:"
        )
    elif task == "bridge":
        query = (
            "For each satellite image, provide a confidence score between 0 and 1 indicating the presence of bridges "
            "(including overpasses and footbridges). Use 1 for definite presence and 0 for absence. Images for analysis:"
        )
    elif task == "car_bridge":
        query = (
            "For each satellite image, provide a confidence score between 0 and 1 indicating the presence of bridges "
            "(including overpasses, excluding footbridges). Use 1 for definite presence and 0 for absence. Images for analysis:"
        )
    elif task == "max_speed":
        query = (
            "For each satellite image, identify the highest speed limit (km/h) on visible roads. "
            "If multiple roads have different limits, report the highest one. If no roads are visible, respond with -100."
            "Base your assessment on road type, lane markings, signage, and infrastructure. Include a brief explanation for each determination."
            "You may estimate any value; it doesn't need to be a multiple of 10. Pay attention to image edges. Images for analysis:"
        )
    elif task == "building_count":
        query = (
            "For each 224x224 pixel satellite image (representing 300x300 meters), count the number of visible buildings. "
            "Focus on rooftops, structures, walls, enclosures, and distinct outlines. Ignore roads, trees, and water bodies. "
            "Provide an accurate count based solely on visible buildings. Analyze each image individually. Images for analysis:"
        )
    else:
        raise ValueError("Task doesn't have a corresponding query")

    model = "gpt-4o"
    # model = "gpt-4o-mini"

    test = False
    make_query(model, task, query, test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="visualizer", description="Query GPT")
    parser.add_argument("output_dir", type=str, help="Path to write pkl file")
    args = parser.parse_args()
    main(args.output_dir)
