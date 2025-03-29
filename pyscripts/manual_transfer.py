import json
import os
import time
from datetime import datetime

import rootutils
import torch
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import LinearSVR, LinearSVC
from torch.utils.data import DataLoader
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.tile_dataset import worker_init_fn

from src.data.components.task_dataset import TaskDataset, collate_tasks
from src.data.components.tiles import TileBatch
from src.modules.embedding_lookup import EmbeddingLookup

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="geojepa_gti_dec30", required=True)
parser.add_argument("--task", type=str, default="building_count", required=True)
args = parser.parse_args()
ckpt = args.ckpt
task = args.task

backbone = EmbeddingLookup(f"data/embeddings/{ckpt}/pretraining_huge", cls_only=True)


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


@torch.no_grad()
def get_data(split: str, limit=None, size="huge"):
    data = TaskDataset(
        f"data/tiles/huge/tasks/{task}", split, size=size, cheat=True, load_images=False
    )
    X_names = []
    X = []
    y = []

    loader = DataLoader(
        data,
        num_workers=16,
        batch_size=32,
        collate_fn=collate_tasks,
        worker_init_fn=worker_init_fn,
    )

    i = 0
    if not limit:
        limit = len(data)
    for batch in tqdm(
        loader, desc=f"Getting {split} data", total=limit // loader.batch_size
    ):
        if i >= limit:
            break
        tiles: TileBatch = batch.tiles
        labels = batch.labels
        embeddings = backbone(tiles).detach().numpy()
        for name, label, emb in zip(tiles.names(), labels, embeddings):
            i += 1
            X_names.append(name)
            X.append(emb.tolist())
            y.append(label.item())

    return X_names, X, y


print(f"Training SVR regressor on {ckpt} embeddings")
_, X_train, y_train = get_data("train")
names_test, X_test, y_test = get_data("test")
names_tiny, X_tiny, y_tiny = get_data("test")


def mae(y, pred):
    return torch.nn.functional.l1_loss(pred, y).item()


def mse(y, pred):
    return torch.nn.functional.mse_loss(pred, y).item()


def eval_model(model, name):
    start = time.time()
    print(f"Fitting {name}...")
    model.fit(X_train, y_train)
    print(f"Predicting {name}...")

    huge = {}
    pred = model.predict(X_test)
    pred = torch.tensor(pred).clamp(
        min=config[task]["min"], max=config[task].get("max", float("inf"))
    )
    y = torch.tensor(y_test)
    huge["mae"] = mae(y, pred)
    huge["mse"] = mse(y, pred)

    tiny = {}
    pred = model.predict(X_tiny)
    pred = torch.tensor(pred).clamp(
        min=config[task]["min"], max=config[task].get("max", float("inf"))
    )
    y = torch.tensor(y_tiny)
    tiny["mae"] = mae(y, pred)
    tiny["mse"] = mse(y, pred)

    total_time = time.time() - start

    date = datetime.today().strftime("%Y-%m-%d")
    res = {
        "model": f"{ckpt}-{name}",
        "dataset": args.task,
        "cheat": False,
        "training_time": total_time,
        "huge": huge,
        "tiny": tiny,
        "date": date,
        "time": datetime.today().strftime("%H:%M:%S"),
    }
    save_dir = f"logs/sklearn/{date}/{task}/{ckpt}/{name}/"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/results.json", "w") as f:
        json.dump(res, f, indent=4)


if task in ["traffic_signals", "building_count", "max_speed"]:
    eval_model(LinearSVR(), "svr")
    eval_model(
        MLPRegressor(
            hidden_layer_sizes=(256,),
            max_iter=2000,
            early_stopping=True,
        ),
        "mlpr",
    )
else:
    eval_model(LinearSVC(), "svc")
    eval_model(
        MLPClassifier(
            hidden_layer_sizes=(256,),
            max_iter=2000,
            early_stopping=True,
        ),
        "mlpc",
    )
