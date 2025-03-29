# -------------------------------
# This files takes a model and a dataset and generates embeddings using the model backbone to speed up training the regression head.
# -------------------------------


from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import rootutils
from tqdm import tqdm

from src.modules.pooling import AvgMaxPool

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
import lightning as L
import torch
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig, OmegaConf
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="lightning.pytorch.trainer.connectors.data_connector",
)

torch.set_float32_matmul_precision("medium")

from src.lightning_utils.utils import get_choice
from src.lightning_utils.resolvers import sweep_name
from src.lightning_utils.instantiators import instantiate_model
from src.utils.data_utils import SimilarSizeBatchDataLoader
from src.data.task_datamodule import TaskBatch

from src.lightning_utils import (
    RankedLogger,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

max_tokens = 0


def high_norm(embeddings):
    norms = torch.norm(embeddings, p=2, dim=2)
    max_norm_indices = torch.argmax(norms, dim=1)
    max_norm_indices = (
        max_norm_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, embeddings.size(2))
    )
    return embeddings.gather(1, max_norm_indices).squeeze(1)


def generate(
    cfg,
    clsh5,
    feath5,
    idsh5,
    loader: SimilarSizeBatchDataLoader,
    model,
    device,
    cls_only,
    start_idx,
    limit,
):
    """
    Generates embeddings and writes them to the preallocated HDF5 datasets.

    :param clsh5: HDF5 file handle for class embeddings.
    :param feath5: HDF5 file handle for features.
    :param idsh5: HDF5 file handle for identifiers.
    :param loader: DataLoader to iterate over.
    :param model: Trained model for generating embeddings.
    :param device: Device to run the model on.
    :param cls_only: Boolean flag to determine if only class embeddings are stored.
    :param start_idx: Starting index for writing into HDF5 datasets.
    :return: Updated index after writing the current batch.
    """
    length = len(loader)
    log.info(
        f"Generating embeddings for {length} batches with batch_size: {loader.batch_size}"
    )

    current_index = start_idx
    total = 0

    avgmaxpool = AvgMaxPool(top_k=20)

    with torch.no_grad():
        for batch in tqdm(loader, total=length, desc="Processing Batches"):
            if total >= limit:
                break
            if isinstance(batch, TaskBatch):
                batch = batch.tiles
            if "16_11268_26027" in batch.names():
                print("Found 16_11268_26027")
                exit(0)
            if "14_2817_6506" in batch.group_names():
                print(
                    "\nFound 14_2817_6506 but not 16, count from group:",
                    len(
                        list(filter(lambda x: x == "14_2817_6506", batch.group_names()))
                    ),
                )
            batch.to(device)
            outputs = model(batch)

            identifiers = batch.names()
            batch_size = batch.size
            if total + batch.size > limit:
                batch_size = limit - total
                identifiers = identifiers[:batch_size]
            total += batch_size

            mods = None

            if isinstance(outputs, Tuple):
                cls = outputs[0]
                features = outputs[1]
                cls = (
                    cls.cpu().numpy().astype("float32")[:batch_size]
                )  # Shape: (batch_size, 768)
                if features is not None:
                    features = (
                        features.cpu().numpy().astype("float32")[:batch_size]
                    )  # Shape: (batch_size, 196, 768)
                if len(outputs) > 2:
                    mods = outputs[2].cpu().numpy().astype("int32")[:batch_size]
                if len(outputs) > 3:
                    sort_indices = outputs[3].cpu().numpy().astype("int32")[:batch_size]
            else:
                features = (
                    outputs.cpu().numpy().astype("float32")[:batch_size]
                )  # Shape: (batch_size, 196, 768)
                cls = None

            # Retrieve identifiers (assuming batch.names() returns a list of strings)
            end_index = current_index + batch_size

            # Write identifiers
            idsh5["identifiers"][current_index:end_index] = identifiers

            def write_feat():
                if feath5 is not None:
                    feature_seq_len = features.shape[1]
                    global max_tokens
                    if feature_seq_len > max_tokens:
                        print(f"New max tokens: {feature_seq_len}")
                        max_tokens = feature_seq_len
                        feath5["features"].resize(max_tokens, axis=1)
                        if mods is not None:
                            feath5["mods"].resize(max_tokens, axis=1)
                    feath5["features"][current_index:end_index, :feature_seq_len, :] = (
                        features
                    )
                    if mods is not None:
                        feath5["mods"][current_index:end_index, :feature_seq_len] = mods

            if cls is not None:
                clsh5["cls"][current_index:end_index] = cls
                if not cls_only and feath5 is not None:
                    write_feat()
            else:
                write_feat()

            current_index = end_index

    return current_index


@task_wrapper
def run(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generates embeddings for the dataset and stores them in HDF5 files.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # if 'ckpt' in cfg.model:
    #     hydra_config_path = cfg.model.ckpt / "hydra" / "config.yaml"

    log.info(f"Instantiating model <{cfg.model._target_}>")
    try:
        model: LightningModule = instantiate_model(
            cfg, overrides={"head": torch.nn.Identity(), "cls_token": False}
        )
    except Exception as e:
        print(
            f"\n\nWARNING: Could not instantiate model with Identity head. Trying with default head. {e}\n\n"
        )
        exit()
        model: LightningModule = instantiate_model(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval().freeze()

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data: LightningDataModule = hydra.utils.instantiate(
        cfg.data, shuffle=False, drop_last=False, pin_memory=False, num_workers=10
    )

    data.setup()

    model_name = get_choice("model")
    data_name = get_choice("data")
    postfix = f"_{cfg.data.size}"
    if cfg.data.cheat:
        postfix += "_cheat"

    save_name = cfg.get("save_name", model_name)
    path = Path(f"data/embeddings/{save_name}/{data_name}{postfix}")
    path.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving embeddings to: {path}")

    cls_path = path / "cls.h5"
    feat_path = path / "feat.h5"
    ids_path = path / "identifiers.h5"  # New dataset for identifiers

    log.info(f"Class Embeddings Path: {cls_path}")
    log.info(f"Feature Embeddings Path: {feat_path}")
    log.info(f"Identifiers Path: {ids_path}")
    log.info(f"Only cls: {cfg.cls_only}")
    log.info(f"Only test: {cfg.test_only}")

    # Calculate total number of images
    total_images = 0
    train_images = 0
    val_images = 0

    test_images = len(data.test_dataloader().dataset)
    if not cfg.test_only:
        train_images = len(data.train_dataloader().dataset)
        val_images = len(data.val_dataloader().dataset)

    if cfg.limit is not None:
        limit = int(cfg.limit)
        train_images = min(limit, train_images)
        val_images = min(limit, val_images)
        test_images = min(limit, test_images)

    total_images = train_images + val_images + test_images

    log.info(f"Total number of images: {total_images}")
    batch = data.test_dataloader().__iter__().__next__().to(device)
    if type(batch) == TaskBatch:
        batch = batch.tiles
    log.info(batch.names())
    tokens = model(batch)
    if isinstance(tokens, Tuple):
        cls_dim = tokens[0].size(-1)
        if cfg.cls == "pool":
            cls_dim *= 2
        if not cfg.cls_only:
            feat_dim = tokens[1].size(-1)
    else:
        cls_dim = tokens.size(-1)
        if cfg.cls == "pool":
            cls_dim *= 2
        feat_dim = 0

    mode = "a" if cfg.append else "w"
    idsh5 = h5py.File(ids_path, mode)
    clsh5 = h5py.File(cls_path, mode)
    feath5 = h5py.File(feat_path, mode) if not cfg.cls_only else None

    try:
        # Create the identifiers dataset as a variable-length string array
        dt = h5py.string_dtype(encoding="utf-8")
        if mode == "w":
            idsh5.create_dataset(
                "identifiers",
                shape=(total_images,),
                dtype=dt,
            )

            if not cfg.cls_only:
                feath5.create_dataset(
                    "features",
                    shape=(total_images, 196, feat_dim),
                    maxshape=(total_images, 197 + 1250 + 1250, feat_dim),
                    dtype="float32",
                )
                feath5.create_dataset(
                    "mods",
                    shape=(total_images, 196),
                    maxshape=(total_images, 197 + 1250 + 1250),
                    dtype="int32",
                )

            clsh5.create_dataset(
                "cls",
                shape=(total_images, cls_dim),
                dtype="float32",
            )

        current_index = 0

        if not cfg.test_only:
            # Generate embeddings for training set
            train_loader = data.train_dataloader()
            current_index = generate(
                cfg,
                clsh5,
                feath5,
                idsh5,
                train_loader,
                model,
                device,
                cfg.cls_only,
                current_index,
                train_images,
            )

            # Generate embeddings for validation set
            val_loader = data.val_dataloader()
            current_index = generate(
                cfg,
                clsh5,
                feath5,
                idsh5,
                val_loader,
                model,
                device,
                cfg.cls_only,
                current_index,
                val_images,
            )

        # Generate embeddings for test set
        test_loader = data.test_dataloader()
        current_index = generate(
            cfg,
            clsh5,
            feath5,
            idsh5,
            test_loader,
            model,
            device,
            cfg.cls_only,
            current_index,
            test_images,
        )

        log.info(
            f"Embedding generation completed. Total images processed: {current_index}"
        )
    finally:
        idsh5.close()
        clsh5.close()
        if feath5 is not None:
            feath5.close()

    return {}, {}


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="backbone-gen-emb.yaml"
)
def main(cfg: DictConfig):
    """
    Main entry point for generating embeddings.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    run(cfg)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("sweep_name", sweep_name)
    main()
