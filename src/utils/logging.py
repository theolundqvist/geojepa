import torch
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from pacmap import PaCMAP
from torch import nn

from src.modules.tokenizer import Modality


def log_emb_image(tb, embedding, name, global_step, plot_sorted=True):
    if tb is None:
        return
    # log.info(f"Logging embedding image {name}")
    print(name)

    def run(emb: torch.Tensor):
        def gen_img(matrix: torch.Tensor):
            matrix = matrix.clone().detach()
            fig, ax = plt.subplots(figsize=(25, 40))
            ax.imshow(matrix.cpu().numpy(), cmap="viridis", interpolation="nearest")
            ax.set_title("Matrix Heatmap using imshow")
            ax.set_xlabel("Hidden dimensions")
            ax.set_ylabel("Tile tokens")
            # patches = [
            #     mpatches.Patch(label=f'cov_loss: {loss.cov_loss.cpu().item()}'),
            #     mpatches.Patch(label=f'var_loss: {loss.std_loss.cpu().item()}'),
            # ]
            # ax.legend(handles=patches, title='Loss', bbox_to_anchor=(1.05, 1), loc='upper left')
            # plt.tight_layout()
            return fig

        fig = gen_img(embedding)
        tb.add_figure(f"emb-heatmap/{name}", fig, global_step)
        plt.close(fig)
        if plot_sorted and emb.shape[0] > 1 and emb.dtype != torch.bool:
            emb = emb.float()
            row_sums = emb.var(dim=1)
            sorted_values, sorted_indices = torch.sort(row_sums, dim=0, descending=True)
            sorted = emb[sorted_indices]
            fig = gen_img(sorted)
            tb.add_figure(f"emb-heatmap-sorted/{name}", fig, global_step)
            plt.close(fig)

    run(embedding)
    # for mod in range(Modality.PAD + 1, Modality.CLS + 1):
    #     run(embedding[mods == mod], Modality(mod).name)


def visualize_representation_space(tb, mods, embeddings, name, global_step):
    if tb is None:
        return

    n = min(10, embeddings.size(0) // 3)
    if n < 1:
        return
    embeddings = embeddings[mods != Modality.PAD, :]
    mods = mods[mods != Modality.PAD]
    pacmap_model = PaCMAP(
        n_components=2, n_neighbors=n, distance="angular", random_state=42
    )
    e = pacmap_model.fit_transform(embeddings.cpu().detach().numpy())
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    labels = list(map(lambda x: x.name, list(Modality)))
    colors = [
        "#ff7f00",  # 0: Orange
        "#1f78b4",  # 1: Blue
        "#33a02c",  # 2: Green
        "#e31a1c",  # 3: Red
        #'#6a3d9a',  # 4: Purple
    ]
    cmap = ListedColormap(colors)
    ax.scatter(
        e[:, 0],
        e[:, 1],
        s=60,
        alpha=0.06,
        cmap=cmap,
        c=mods.cpu().numpy(),
        vmin=float(Modality.PAD),
        vmax=float(len(Modality)),
    )
    ax.set_axis_off()
    ax.set_title("Embedding PaCMAP")
    patches = [
        mpatches.Patch(color=colors[i], label=f"{labels[i]}")
        for i in range(len(labels))
    ]
    ax.legend(handles=patches, title="Mods", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    tb.add_figure(f"PaCMAP/{name}", fig, global_step)
    plt.close(fig)


def get_weight_statistics(model: nn.Module, require_grad: bool = False) -> dict:
    """
    Computes the mean, max, and min of all weights in the given nn.Module.

    Args:
        model (nn.Module): The neural network model.

    Returns:
        dict: A dictionary containing the mean, max, and min of all weights.
    """
    all_weights = []

    # Iterate through all parameters in the model
    for name, param in model.named_parameters():
        if param.requires_grad or not require_grad:
            # Flatten the parameter tensor and add to the list
            all_weights.append(param.data.view(-1))
            # print(f"Collected weights from layer: {name}, shape: {param.shape}")

    if not all_weights:
        raise ValueError("No trainable parameters found in the model.")

    # Concatenate all weights into a single tensor
    concatenated_weights = torch.cat(all_weights)

    # Compute statistics
    mean_val = torch.mean(concatenated_weights).item()
    max_val = torch.max(concatenated_weights).item()
    min_val = torch.min(concatenated_weights).item()
    std_val = torch.std(concatenated_weights).item()

    return {"mean": mean_val, "max": max_val, "min": min_val, "std": std_val}
