import json
import os
import time
from datetime import datetime

import rootutils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.tile_dataset import worker_init_fn
from src.data.components.task_dataset import TaskDataset, collate_tasks
from src.data.components.tiles import TileBatch
from src.modules.embedding_lookup import EmbeddingLookup

import argparse

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt",
    type=str,
    default="data/embeddings/geojepa_gti_dec30",
    required=False,
    help="Checkpoint name (default: geojepa_gti_dec30)",
)
parser.add_argument(
    "--task", type=str, default="all", required=False, help="Task name (default: all)"
)
args = parser.parse_args()
ckpt = args.ckpt
ckpt = ckpt.replace("data/embeddings/", "")

# Initialize Backbone

# Configuration for tasks
config = {
    "car_bridge": {
        "type": "classification",
        "min": 0,
        "max": 1,
    },
    "traffic_signals": {
        "type": "regression",
        "beta": 0.5,
        "min": 0,
    },
    "max_speed": {
        "type": "regression",
        "beta": 1.0,
        "min": -100,
    },
    "building_count": {
        "type": "regression",
        "beta": 5.0,
        "min": 0,
    },
    "bridge": {
        "type": "classification",
        "min": 0,
        "max": 1,
    },
}

emb_task_dir = "pretraining_huge"
if args.task != "all":
    config = {args.task: config[args.task]}
    emb_task_dir = f"{args.task}_huge"

backbone = EmbeddingLookup(f"data/embeddings/{ckpt}/{emb_task_dir}", cls_only=True)


@torch.no_grad()
def get_data(task: str, split: str, limit=None, size="huge"):
    data = TaskDataset(
        f"data/tiles/{size}/tasks/{task}",
        split,
        size=size,
        cheat=True,
        load_images=False,
    )
    X_names = []
    X = []
    y = []

    loader = DataLoader(
        data,
        num_workers=0 if data == "small" else 16,
        batch_size=32,
        collate_fn=collate_tasks,
        worker_init_fn=worker_init_fn,
    )

    i = 0
    if limit is None:
        limit = len(data)
    total_steps = min(len(loader), (limit + loader.batch_size - 1) // loader.batch_size)

    for batch in tqdm(
        loader, desc=f"Getting {split} data for {task}", total=total_steps
    ):
        if i >= limit:
            break
        tiles: TileBatch = batch.tiles
        labels = batch.labels
        try:
            embeddings = backbone(tiles).detach()
        except Exception:
            continue
        for name, label, emb in zip(tiles.names(), labels, embeddings):
            if i >= limit:
                break
            i += 1
            X_names.append(name)
            X.append(emb.tolist())
            y.append(label.item())

    return (
        X_names,
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


print(f"Preparing data using checkpoint: {ckpt}")
tasks = config.keys()

# Define Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Metric Computation Functions
def compute_regression_metrics(y_true, y_pred):
    mae = nn.functional.l1_loss(y_pred, y_true).item()
    mse = nn.functional.mse_loss(y_pred, y_true).item()
    return {"mae": mae, "mse": mse}


def compute_classification_metrics(y_true, y_pred):
    # y_pred should be binary (0 or 1)
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"accuracy": accuracy, "f1": f1}


# Evaluation Function
@torch.no_grad()
def eval_model(model, dataloader, task_type, min_val, max_val):
    model.eval()
    all_preds = []
    all_labels = []
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        outputs = model(X_batch)
        outputs = torch.clamp(outputs, min=min_val, max=max_val)
        if task_type == "regression":
            pass
        elif task_type == "classification":
            outputs = torch.sigmoid(outputs).round()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        all_preds.append(outputs)
        all_labels.append(y_batch)

    if len(all_preds) == 0:
        return {}
    if len(all_labels) == 0:
        return {}
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    metrics = compute_regression_metrics(labels, preds)
    if task_type == "classification":
        metrics.update(compute_classification_metrics(labels, preds))
    return metrics


# Iterate over each task and train/evaluate models
for task, task_config in config.items():
    print(f"\nProcessing task: {task}")
    task_type = task_config["type"]

    # Retrieve Data
    _, X_train, y_train = get_data(task, "train")
    _, X_val, y_val = get_data(task, "val")
    _, X_test, y_test = get_data(task, "test")
    _, X_small, y_small = get_data(
        task, "test", size="small"
    )  # Assuming 'small' split exists

    # Create Datasets and DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    small_dataset = TensorDataset(X_small, y_small)

    train_loader = DataLoader(
        train_dataset, batch_size=1024, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    small_loader = DataLoader(small_dataset, batch_size=1024, shuffle=False)

    # Define Models
    input_dim = X_train.shape[1]

    if task_type == "regression":
        # Linear Regression Model
        class LinearRegressionModel(nn.Module):
            def __init__(self, input_dim):
                super(LinearRegressionModel, self).__init__()
                self.linear = nn.Linear(input_dim, 1)

            def forward(self, x):
                return self.linear(x).squeeze(1)

        # MLP Regression Model
        class MLPRegressionModel(nn.Module):
            def __init__(self, input_dim, hidden_dims=(512,)):
                super(MLPRegressionModel, self).__init__()
                layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(0.1))
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, 1))
                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x).squeeze(1)

        models = {
            f"{ckpt}_linr": LinearRegressionModel(input_dim).to(device),
            f"{ckpt}_mlpr": MLPRegressionModel(input_dim).to(device),
        }
        loss_fn = nn.SmoothL1Loss(beta=float(task_config.get("beta", 1.0)))

    elif task_type == "classification":
        # Linear Classification Model
        class LinearClassificationModel(nn.Module):
            def __init__(self, input_dim):
                super(LinearClassificationModel, self).__init__()
                self.linear = nn.Linear(input_dim, 1)

            def forward(self, x):
                return self.linear(x).squeeze(1)

        # MLP Classification Model
        class MLPClassificationModel(nn.Module):
            def __init__(self, input_dim, hidden_dims=(512,)):
                super(MLPClassificationModel, self).__init__()
                layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(0.1))
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, 1))
                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x).squeeze(1)

        models = {
            f"{ckpt}_linc": LinearClassificationModel(input_dim).to(device),
            f"{ckpt}_mlpc": MLPClassificationModel(input_dim).to(device),
        }
        loss_fn = nn.BCEWithLogitsLoss()

    else:
        print(f"Unknown task type: {task_type}. Skipping...")
        continue

    # Training Parameters
    epochs = 200
    learning_rate = 1e-3

    # Iterate over each model type
    min_pred = task_config.get("min", float("-inf"))
    max_pred = task_config.get("max", float("inf"))
    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=10
        )
        # scheduler = optim.lr_scheduler.SequentialLR(optimizer, milestones=[int(epochs * 0.2)],
        #                                 schedulers=[
        #                                     optim.lr_scheduler.ConstantLR(optimizer, learning_rate),
        #                                     optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        #                                 ])

        best_loss = float("inf")
        patience = 35
        trigger_times = 0

        best_model_state = None  # Initialize to store the best model

        train_loss = float("inf")
        epoch = 0
        pbar = tqdm(range(1, epochs + 1), total=epochs)
        start = time.time()
        for epoch in pbar:
            model.train()
            running_loss = 0.0
            running_mae = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                if task_type == "regression":
                    # outputs = torch.clamp(outputs, min=min_pred, max=max_pred)
                    loss = loss_fn(outputs, y_batch)
                elif task_type == "classification":
                    loss = loss_fn(outputs, y_batch)

                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X_batch.size(0)
                running_mae += nn.functional.l1_loss(outputs, y_batch).item()

            train_loss = running_loss / len(train_loader.dataset)
            train_mae = running_mae / len(train_loader)

            running_loss = 0
            running_mae = 0
            running_unique_count = 0
            with torch.no_grad():
                model.eval()
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    outputs = model(X_batch)
                    outputs = torch.clamp(outputs, min=min_pred, max=max_pred)
                    if task_type == "regression":
                        loss = loss_fn(outputs, y_batch)
                    elif task_type == "classification":
                        loss = loss_fn(outputs, y_batch)
                    running_loss += loss.item() * X_batch.size(0)
                    running_mae += nn.functional.l1_loss(outputs, y_batch).item()
                    running_unique_count += outputs.unique(dim=0).size(0)
            val_loss = running_loss / len(val_loader.dataset)
            val_mae = running_mae / len(val_loader)
            avg_unique_count = running_unique_count / len(val_loader)

            pbar.set_postfix_str(
                f"Epoch {epoch}/{epochs} train_mae: {train_mae:.4f}, val_mae: {val_mae:.4f} train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}"
            )

            # Step the scheduler
            lr = scheduler.get_last_lr()[0]
            scheduler.step(val_loss)
            new_lr = scheduler.get_last_lr()[0]
            if lr != new_lr:
                print(f"New lr: {lr} -> {new_lr}")

            # Early Stopping
            if val_loss < best_loss:
                best_loss = val_loss
                trigger_times = 0
                # Save the best model
                best_model_state = model.state_dict()
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("Early stopping!")
                    break

        if best_model_state is not None:
            # Load the best model
            model.load_state_dict(best_model_state)
        else:
            print("No improvement during training.")

        # Evaluation
        print("Evaluating on test set...")
        test_metrics = eval_model(
            model, test_loader, task_type, min_val=min_pred, max_val=max_pred
        )
        print(test_metrics)

        print("Evaluating on small set...")
        small_metrics = eval_model(
            model, small_loader, task_type, min_val=min_pred, max_val=max_pred
        )
        print(small_metrics)

        # Prepare results
        date = datetime.today().strftime("%Y-%m-%d")
        current_time = datetime.today().strftime("%H:%M:%S")
        res = {
            "model": f"{model_name}",
            "dataset": task,
            "cheat": True,
            "training_time": time.time() - start,
            "training_epochs": epoch,
            "best_train_loss": best_loss,
            "huge": test_metrics,
            "small": small_metrics,
            "date": date,
            "time": current_time,
        }

        # Save Results
        save_dir = f"logs/pytorch/{date}/{task}/{ckpt}/{model_name}/"
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/results.json", "w") as f:
            json.dump(res, f, indent=4)
        print(f"Results saved to {save_dir}/results.json")
