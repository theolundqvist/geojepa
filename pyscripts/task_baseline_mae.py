import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch


def get_parent_tile(new_x, new_y) -> str:
    zoom_diff = 16 - 14
    old_x = new_x // (2**zoom_diff)
    old_y = new_y // (2**zoom_diff)
    return f"14_{old_x}_{old_y}"


def parse_tile_id(tile_id: str) -> Tuple[int, int]:
    # Parse tile_id assuming format "x_y". Adjust if format differs.
    parts = tile_id.split("_")
    return int(parts[1]), int(parts[2])


def main():
    parser = argparse.ArgumentParser(
        description="Summary statistics for label and feature count files."
    )
    parser.add_argument("task", type=str, help="Path to the task dir")
    args = parser.parse_args()
    task_dir = Path(args.task)
    label_file = task_dir / "labels.txt"
    splits_file = task_dir / "../../logs/file_split_log.txt"

    # Read labels
    labels = pd.read_csv(label_file, sep=":", header=None, names=["tile_id", "label"])
    labels["tile_id"] = labels["tile_id"].astype(str)
    labels["parent_tile"] = labels["tile_id"].apply(
        lambda tid: get_parent_tile(*parse_tile_id(tid))
    )
    labels = labels.set_index("tile_id")

    small_labels = pd.read_csv(
        Path(args.task.replace("huge", "small")) / "labels.txt",
        sep=":",
        header=None,
        names=["tile_id", "label"],
    )
    small_labels["tile_id"] = small_labels["tile_id"].astype(str)
    small_labels = small_labels.set_index("tile_id")

    splits = pd.read_csv(
        splits_file,
        sep=" -> ",
        engine="python",
        header=None,
        names=["tile_id", "split"],
    ).set_index("tile_id")

    labels_dict = labels.to_dict()["label"]
    small_labels_dict = small_labels.to_dict()["label"]

    parent_tile_dict = labels.to_dict()["parent_tile"]
    split_dict = splits.to_dict()["split"]

    train_values = []
    test_values = []
    small_test_values = []

    for tile_id, label in labels_dict.items():
        if tile_id in parent_tile_dict:
            parent_tile = parent_tile_dict[tile_id]
            if parent_tile in split_dict:
                if split_dict[parent_tile] == "train":
                    train_values.append(label)
                elif split_dict[parent_tile] == "test":
                    test_values.append(label)

    for tile_id, label in small_labels_dict.items():
        if tile_id in parent_tile_dict:
            parent_tile = parent_tile_dict[tile_id]
            if parent_tile in split_dict:
                if split_dict[parent_tile] == "test":
                    small_test_values.append(label)

    print(f"Train values: {len(train_values)}")
    print(f"Test values: {len(test_values)}")
    print(f"Small test values: {len(small_test_values)}")

    train = torch.tensor(train_values)
    test = torch.tensor(test_values)
    small_test = torch.tensor(small_test_values)

    # # avg
    #     avg_train = train.mean().item()
    #     avg_test = test.mean().item()
    #     print(f"Average train value: {avg_train}")
    #     print(f"Average test value: {avg_test}")
    #     mae_train = (train - avg_train).abs().mean().item()
    #     mae_test = (test - avg_test).abs().mean().item()
    #     print(f"MAE train (avg): {mae_train}")
    #     print(f"MAE test (avg): {mae_test}")
    #
    #     #mode
    #     mode_train = train.mode().values.item()
    #     mode_test = test.mode().values.item()
    #     print(f"Mode train value: {mode_train}")
    #     print(f"Mode test value: {mode_test}")
    #     mae_train = (train - mode_train).abs().mean().item()
    #     mae_test = (test - mode_test).abs().mean().item()
    #     print(f"MAE train (mode): {mae_train}")
    #     print(f"MAE test (mode): {mae_test}")

    # median
    median_train = train.median().item()
    median_test = test.median().item()
    small_median = small_test.median().item()
    print(f"Median train value: {median_train}")
    print(f"Median test value: {median_test}")
    mae_train = (train - median_train).abs().mean().item()
    mae_test = (test - median_train).abs().mean().item()
    mae_small = (small_test - median_train).abs().mean().item()
    print(f"Task: {args.task.split('/')[-1]}")
    print(f"train MAE (train median): {mae_train:.2f}")
    print(f"test MAE (train median): {mae_test:.2f}")
    print(f"small test MAE (train median): {mae_small:.2f}")


if __name__ == "__main__":
    main()
