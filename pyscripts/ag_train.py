import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from autogluon.multimodal import MultiModalPredictor
import json

script_path = os.path.dirname(os.path.realpath(__file__))
log_dir = script_path + "/../ag-logs/"
os.makedirs(log_dir + "/run-logs", exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            log_dir + "/run-logs/" + datetime.today().strftime("%Y-%m-%d_%H:%M:%S")
        ),
        logging.StreamHandler(),
    ],
)


def load_split(temp_dir, tags, images):
    name = "ag"
    if tags:
        name += "_tags"
    if images:
        name += "_images"

    return pd.read_csv(temp_dir + f"/{name}_tiling.csv")


config = {
    "traffic_signals": {
        "min": 0,
    },
    "building_count": {
        "min": 0,
    },
    "max_speed": {
        "min": -100,
    },
    "bridge": {
        "min": 0,
        "max": 1,
    },
    "car_bridge": {
        "min": 0,
        "max": 1,
    },
}


def save_eval(task, predictor, df: pd.DataFrame, name):
    eval_dir = Path(f"{log_dir}/eval/{name}")
    eval_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Evaluating model {name}...")
    res = predictor.predict(df, as_pandas=False)
    res = (
        torch.tensor(res)
        .float()
        .clamp(min=config[task]["min"], max=config[task].get("max", float("inf")))
    )
    df["pred"] = res
    mse = torch.nn.functional.mse_loss(
        torch.tensor(df["pred"]), torch.tensor(df["label"])
    ).item()
    mae = torch.nn.functional.l1_loss(
        torch.tensor(df["pred"]), torch.tensor(df["label"])
    ).item()
    df = df[["subtile", "label", "pred"]]
    df.to_pickle(eval_dir / "predictions.pkl")
    df.to_csv(eval_dir / "predictions.csv")
    with open(eval_dir / "score.txt", "w") as f:
        f.write("Test score:\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"MAE: {mae}\n")
        f.write("First 10 zero labels:\n")
        f.write(str(df[df["label"] == 0][:10]) + "\n")
        f.write("First 10 non-zero labels:\n")
        f.write(str(df[df["label"] != 0][:10]) + "\n")
    return {
        "test/mae": mae,
        "test/mse": mse,
    }


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--time-limit", type=int, default=300)
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--quality", type=str, default="medium")
    p.add_argument("--tags", action="store_true")
    p.add_argument("--images", action="store_true")
    args = p.parse_args()
    huge = script_path + "/../data/autogluon/huge"
    small = script_path + "/../data/autogluon/small"

    if not os.path.exists(huge + "/" + args.task):
        raise ValueError(f"Task {args.task} not found in {huge}")

    if not os.path.exists(small + "/" + args.task):
        raise ValueError(f"Task {args.task} not found in {small}")

    # Create a split of the data
    logger.info("Loading")
    val_df = load_split(f"{huge}/{args.task}/val", args.tags, args.images)
    train_df = load_split(f"{huge}/{args.task}/train", args.tags, args.images)
    hp = {}
    if args.tags:
        hp["env.per_gpu_batch_size"] = 2

    # Initialize the ImagePredictor for regression
    predictor = MultiModalPredictor(
        label="label", problem_type="regression", hyperparameters=hp
    )

    today = datetime.today().strftime("%Y-%m-%d")
    name = f"{today}/{args.task}/{args.quality}-{args.time_limit}s"
    (Path(log_dir) / "eval" / name).mkdir(parents=True, exist_ok=True)

    logging.info("Training the model...")
    preset = args.quality + "_quality"

    start = time.time()
    predictor.fit(
        train_df,
        tuning_data=val_df,
        time_limit=args.time_limit,
        presets=preset,
    )
    total_time = time.time() - start
    logging.info(f"Training took {total_time} seconds")

    test_df = load_split(f"{huge}/{args.task}/test", args.tags, args.images)
    small_test_df = load_split(f"{small}/{args.task}/test", args.tags, args.images)

    logging.info(predictor.fit_summary())
    with open(f"{log_dir}/eval/{name}/fit_summary.txt", "w") as f:
        f.write(str(predictor.fit_summary()))

    huge = save_eval(args.task, predictor, test_df, f"{name}/huge")
    small = save_eval(args.task, predictor, small_test_df, f"{name}/small")
    postfix = ""
    if args.tags:
        postfix += "T"
    if args.images:
        postfix += "I"
    res = {
        "model": f"ag-{postfix}-{args.quality}-{args.time_limit}s",
        "dataset": args.task,
        "cheat": False,
        "training_time": total_time,
        "huge": huge,
        "small": small,
        "date": datetime.today().strftime("%Y-%m-%d"),
        "time": datetime.today().strftime("%H:%M:%S"),
    }
    with open(f"{log_dir}/eval/{name}/results.json", "w") as f:
        json.dump(res, f, indent=4)

    # predictor.save(f"{name}/model.ag")


if __name__ == "__main__":
    main()
