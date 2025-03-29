#!/usr/bin/env python

from rootutils import rootutils
from torch import Tensor
from tqdm import tqdm

from h5_read import FakeTileBatch
from src.data.components.raw_tile_dataset import load_images_from_path
from src.modules.masks import apply_mask

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.modules.tag_encoder import TagIndexer
from src.modules.tokenizer import Modality, compute_normalized_bboxes

from typing import List

import torch
import numpy as np
from PIL import Image, ImageDraw
import logging

import base64
import matplotlib.pyplot as plt
from io import BytesIO

from dash import dcc, html, Input, Output, no_update, Dash, State
import plotly.graph_objects as go

from src.modules.embedding_lookup import EmbeddingLookup
from src.data.components.tile_dataset import TileDataset
from src.data.components.tiles import Tile

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# Example usage in your dataset code. Real code needs the HDF5 “group = file[str(idx)]”.
# We'll define a stub that just returns random data:
class TileReader:
    def __init__(self, task="pretraining", split="test", size="huge", cheat=True):
        self.dataset = TileDataset(
            f"data/tiles/{size}/tasks/{task}",
            split,
            cheat=cheat,
            use_image_transforms=False,
            load_images=False,
            add_original_image=False,
        )
        self.dataset.__init_worker__()

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def __len__(self):
        return self.dataset.__len__()


######################################################
# DRAWING BOUNDING BOXES, CONVERTING TO BASE64 FOR PLOT
######################################################


def draw_bboxes_on_image(tile: Tile, min_box: torch.Tensor):
    """
    Draw bounding boxes on a copy of sat_img. min_boxes is shape [F, 4, 2].
    sat_img is [3, H, W].
    """
    # Convert the Torch image to a [H, W, 3] numpy
    # Here, assume it's 0..1 float in each pixel
    img = load_images_from_path(f"data/tiles/huge/images/{tile.group_name()}.webp")[
        (tile.tile_coord[1].item(), tile.tile_coord[2].item())
    ]
    np_img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(np_img)

    draw = ImageDraw.Draw(pil_img)
    # box shape is [4,2], corners in XY.
    # We'll assume XY is (x,y) in pixel coords
    coords = list(
        (min_box * torch.tensor((pil_img.width, pil_img.height))).reshape(-1).tolist()
    )
    draw.polygon(coords, outline="red", width=2)

    return pil_img


def pil_to_base64(pil_img: Image.Image):
    """
    Convert a PIL image to a base64-encoded string suitable for <img src='...' />.
    """
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def compute_box_area(box):
    x_coords = box[:, 0]
    y_coords = box[:, 1]
    width = torch.max(x_coords) - torch.min(x_coords)
    height = torch.max(y_coords) - torch.min(y_coords)
    return width * height


def crop_image_to_box(img: Image, box):
    x_coords = box[:, 0]
    y_coords = box[:, 1]

    left = torch.min(x_coords).clamp(0, 1).item() * img.width
    top = torch.min(y_coords).clamp(0, 1).item() * img.height
    right = torch.max(x_coords).clamp(0, 1).item() * img.width
    bottom = torch.max(y_coords).clamp(0, 1).item() * img.height

    crop_box = (left, top, right, bottom)
    return img.crop(crop_box)


def get_in_mod_index(tile_mods, token_id):
    mod = tile_mods[token_id].item()
    first_mod_index = list(tile_mods).index(mod)
    if first_mod_index == -1:
        print("WARNING: feature not found")
    in_mod_token_idx = token_id - first_mod_index
    assert in_mod_token_idx >= 0
    return in_mod_token_idx


def get_min_box(tile, tile_mods, token_id):
    mod = tile_mods[token_id].item()
    in_mod_token_idx = get_in_mod_index(tile_mods, token_id)

    min_box = None
    if mod == Modality.OSM:
        min_box = tile.min_boxes[in_mod_token_idx]
    elif mod == Modality.IMG:
        if in_mod_token_idx == 0:
            return torch.tensor(((0, 0), (1, 0), (1, 1), (0, 1))).reshape(4, 2)
        min_box = compute_normalized_bboxes(1, 14, device="cpu")[0][
            in_mod_token_idx - 1
        ]
        min_box = min_box.reshape(4, 2)
    assert min_box is not None
    return min_box


def decode_tags(tag_tensor: torch.Tensor, indexer: TagIndexer) -> List[str]:
    """
    Stub function to convert numeric tag IDs into strings.
    e.g. tag_tensor is shape [T], each an ID. Return a list of tag strings.
    Real code might map IDs -> strings from a vocabulary.
    """
    tags = [indexer.index_to_tag(i) for i in tag_tensor]
    return [f"{k}={v}" for k, v in tags if k != "PAD"]


def compute_areas(bboxes):
    """
    Computes the area of each bounding box using the Shoelace formula.
    Args:
        bboxes (torch.Tensor): Tensor of shape (B, 4, 2) representing B bounding boxes.
                               Each bounding box has 4 vertices with (x, y) coordinates.
    Returns:
        torch.Tensor: Tensor of shape (B,) containing the area of each bounding box.
    """
    B, N, C = bboxes.shape
    assert N == 4 and C == 2, "Bounding boxes should have shape (B, 4, 2)"

    bboxes_closed = torch.cat([bboxes, bboxes[:, :1, :]], dim=1)  # Shape: (B, 5, 2)

    x = bboxes_closed[:, :, 0]  # Shape: (B, 5)
    y = bboxes_closed[:, :, 1]  # Shape: (B, 5)

    # Compute (x_i * y_{i+1}) and (x_{i+1} * y_i)
    sum1 = torch.sum(x[:, :-1] * y[:, 1:], dim=1)
    sum2 = torch.sum(y[:, :-1] * x[:, 1:], dim=1)

    # Area is half the absolute difference
    area = 0.5 * torch.abs(sum1 - sum2)

    return area


######################################################
# MAIN PLOTTING LOGIC: REDUCE EMBEDDINGS + MAKE PLOTLY
######################################################
def create_geometry_image(tile: Tile, fid: int):
    fig, ax = plt.subplots()
    coords = feature.points

    if feature.is_polygon:
        polygon = plt.Polygon(coords, closed=True, fill=True, color="lightblue")
        ax.add_patch(polygon)
    elif feature.is_line:
        x, y = zip(*coords)
        ax.plot(x, y, color="blue", linewidth=2)
    elif feature.is_relation_part:
        x, y = zip(*coords)
        ax.plot(x, y, color="blue")

    ax.set_aspect("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")  # Hide axes

    # Save to a bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def main(args):
    n = args.n

    tag_indexer = TagIndexer()

    data = TileReader(args.task, args.task_split, args.task_size, args.task_cheat)
    tiles: List[Tile] = []

    embedding_model = EmbeddingLookup(args.embeddings_dir, mods=True, cls_only=False)
    embeddings: List[torch.Tensor] = []
    gid_to_tile: List[int] = []
    local_token_id: List[int] = []
    modalities: List[int] = []
    tile_mods: List[List[int]] = []
    min_boxes: List[torch.Tensor] = []
    print(f"Reading {n} tiles and embeddings...")
    for tile_id, tile in tqdm(enumerate(iter(data))):
        if len(tiles) >= n:
            break
        if tile.name() not in embedding_model.id_map:
            continue
        tiles.append(tile)
        cls, feats, mods = embedding_model(FakeTileBatch(tile.name()))
        mods: Tensor = mods
        feats: Tensor = apply_mask(feats, mods != Modality.PAD)
        mods: Tensor = apply_mask(mods, mods != Modality.PAD)
        # only first batch
        feats = feats[0]
        mods = mods[0]
        # cls = cls[0]
        for token_id, token in enumerate(feats):
            gid_to_tile.append(tile_id)
            local_token_id.append(token_id)
            embeddings.append(token.unsqueeze(0))
            modalities.append(mods[token_id])
            min_boxes.append(get_min_box(tile, mods, token_id).unsqueeze(0))
        tile_mods.append(mods)

    modalities = torch.tensor(modalities)
    unique_mods = list(modalities.unique())

    embeddings: torch.Tensor = torch.cat(embeddings, dim=0)
    min_boxes: torch.Tensor = torch.cat(min_boxes, dim=0)
    N, C = embeddings.shape
    log.info(f"Loaded {N} total tokens from {n} tiles.")

    # from sklearn.decomposition import PCA
    from pacmap import PaCMAP

    pacmap_model = PaCMAP(n_components=2)
    embeddings_2d = pacmap_model.fit_transform(embeddings.numpy())

    colors = compute_areas(min_boxes).numpy()
    print(modalities.shape, modalities)
    colors = (modalities * 10).numpy()

    # Create the scatter plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            mode="markers",
            marker=dict(size=10, color=colors),
        )
    )

    # Initialize the Dash app
    app = Dash(__name__)

    # Define the layout of the app
    app.layout = html.Div(
        className="container",
        children=[
            dcc.Store(id="color-state", data=False),
            dcc.Graph(id="graph", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="hover-tooltip", direction="bottom"),
            html.Button(
                "Color [default]",
                id="toggle-color-btn",
                n_clicks=0,
                style={"height": "40px", "margin-left": "20px"},
            ),
        ],
        # style={'display': 'flex', 'align-items': 'center'}
    )

    # Update the hover display function to use the new images
    @app.callback(
        Output("hover-tooltip", "show"),
        Output("hover-tooltip", "bbox"),
        Output("hover-tooltip", "children"),
        Input("graph", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        point_idx = hover_data["pointNumber"]
        tile_id = gid_to_tile[point_idx]
        tile = tiles[tile_id]
        token_id = local_token_id[point_idx]
        mod = modalities[point_idx].item()
        mod_name = Modality(mod).name
        min_box = min_boxes[point_idx]
        in_mod_token_idx = get_in_mod_index(tile_mods[tile_id], token_id)

        tile_img = draw_bboxes_on_image(tile, min_box)
        tile_img_base64 = pil_to_base64(tile_img)

        elements = [
            html.P(f"{mod_name}, token_idx: {token_id}", style={"font-weight": "bold"}),
            html.P("Tile Img", style={"font-weight": "bold"}),
            html.Img(
                src=tile_img_base64,  # Use the image generated for hover
                style={
                    "width": "100px",
                    "height": "100px",
                    "display": "block",
                    "margin": "0 auto",
                },
            ),
        ]

        if compute_box_area(min_box) < 0.9:
            token_img = crop_image_to_box(tile_img, min_box)
            token_img = pil_to_base64(token_img)
            elements.append(html.P("Token Img", style={"font-weight": "bold"}))
            elements.append(
                html.Img(
                    src=token_img,  # Use the image generated for hover
                    style={
                        "max-width": "100px",
                        "max-height": "100px",
                        "flex": 1,
                        "display": "block",
                        "margin": "0 auto",
                    },
                )
            )

        # if mod == Modality.GEO:
        #     elements.append(html.P(f"Geometry", style={"font-weight": "bold"}))
        #     elements.append(html.Img(
        #         src=create_geometry_image(tile, token_id - first_mod_index),
        #         style={"width": "100px", "height": "100px", "display": "block", "margin": "0 auto"},
        #     ))

        if mod == Modality.OSM:
            tags = decode_tags(tile.tags[in_mod_token_idx], tag_indexer)
            elements.append(html.P("Tags", style={"font-weight": "bold"}))
            for tag in tags:
                elements.append(html.P(tag))

        return True, bbox, [html.Div(elements)]

    @app.callback(
        [
            Output("graph", "figure"),
            Output("color-toggle-btn", "children"),
            Output("color-state", "data"),
        ],
        [Input("color-toggle-btn", "n_clicks")],
        [State("color-state", "data")],
    )
    def toggle_highlight(n_clicks, toggle_state):
        # Toggle the state
        new_state = not toggle_state

        # Update button text based on the new state
        button_text = "Disable Highlight" if new_state else "Enable Highlight"

        # Update the plot based on the new state
        # if new_state:
        #     # Example setting: Highlight a specific species
        #     highlighted_species = 'setosa'
        #     fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species',
        #                      title='Iris Dataset Scatter Plot - Highlight Setosa')
        #     # Add a shape or annotation to highlight
        #     fig.add_shape(
        #         type="rect",
        #         x0=df[df['species'] == highlighted_species]['sepal_width'].min(),
        #         y0=df[df['species'] == highlighted_species]['sepal_length'].min(),
        #         x1=df[df['species'] == highlighted_species]['sepal_width'].max(),
        #         y1=df[df['species'] == highlighted_species]['sepal_length'].max(),
        #         line=dict(color="RoyalBlue"),
        #         fillcolor="LightSkyBlue",
        #         opacity=0.3,
        #         layer="below"
        #     )
        # else:
        #     # Default plot without highlights
        #     fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species',
        #                      title='Iris Dataset Scatter Plot')

        return fig, button_text, new_state

    app.run_server(debug=False, threaded=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot embeddings with hover-over images."
    )
    parser.add_argument("--n", type=int, default=50, help="Number of tiles to plot")
    parser.add_argument(
        "--task", type=str, default="pretraining", help="Task directory"
    )
    parser.add_argument("--task-size", type=str, default="huge", help="Task size")
    parser.add_argument("--task-split", type=str, default="test", help="Task split")
    parser.add_argument("--task-cheat", type=bool, default=True, help="Task cheat")
    parser.add_argument(
        "-i", "--embeddings_dir", type=str, required=True, help="Embeddings directory"
    )
    args = parser.parse_args()
    if args.task == "pretraining":
        args.task_cheat = False
    main(args)
