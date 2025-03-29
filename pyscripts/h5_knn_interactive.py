import argparse
import time

import cv2
import h5py
import rootutils
from pacmap import PaCMAP
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
from src.data.components.raw_tile_dataset import load_images_from_path
import dash_bootstrap_components as dbc

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import base64


# Define your existing functions here
def get_parent_tile(x: int, y: int, current_z: int = 16, parent_z: int = 14):
    assert current_z > parent_z, "Current zoom must be greater than parent zoom."
    zoom_diff = current_z - parent_z
    parent_x = x // (2**zoom_diff)
    parent_y = y // (2**zoom_diff)
    return parent_x, parent_y


def load_embeddings(cls_path, feat_path=None):
    embeddings_dict = {}
    use_features = False

    id_path = cls_path.replace("cls.h5", "identifiers.h5")
    with h5py.File(id_path, "r") as f:
        ids = f["identifiers"][:]

    if isinstance(ids[0], bytes):
        ids = [id.decode("utf-8") for id in ids]
    else:
        ids = [str(id) for id in ids]

    id_map = {tile_name: idx for idx, tile_name in enumerate(ids)}
    del ids

    if feat_path and os.path.exists(feat_path):
        with h5py.File(feat_path, "r") as feat_h5:
            print(f"Loading features from {feat_path}")
            for group_name in feat_h5:
                embeddings_dict[group_name] = feat_h5[group_name]["features"][:]
                print(
                    f"  features data shape: {embeddings_dict[group_name].shape}: {group_name}"
                )
                print(
                    f"  features data shape: {embeddings_dict[group_name][1].shape}: {group_name}"
                )
        use_features = True
    else:
        with h5py.File(cls_path, "r") as cls_h5:
            print(f"Loading cls from {cls_path}")
            for tile_name, id in id_map.items():
                embeddings_dict[tile_name] = cls_h5["cls"][id][:]

    if embeddings_dict:
        first_key = next(iter(embeddings_dict))
        embedding_dim = embeddings_dict[first_key].shape[-1]
        print(f"Embedding dimension: {embedding_dim}")
    else:
        raise ValueError("No embeddings found in the provided HDF5 files.")

    return embeddings_dict, embedding_dim


def compute_embeddings(embeddings_dict):
    group_names = list(embeddings_dict.keys())
    all_embeddings = np.stack([embeddings_dict[name] for name in group_names])
    pacmap_model = PaCMAP(n_components=2)
    reduced = pacmap_model.fit_transform(all_embeddings)
    return group_names, all_embeddings, reduced


def load_images(tile_names, image_dir, default_image_path, zoom=0.25):
    """
    Load the image corresponding to a group name.

    Parameters:
    - group_name (str): The group name.
    - image_dir (str): Directory where images are stored.
    - default_image_path (str): Path to a default image if group image is missing.
    - zoom (float): Scaling factor for the image.

    Returns:
    - image (OffsetImage): The image to embed in the plot.
    """
    image_dict = {}
    for tile_name in tqdm(tile_names, desc="Loading images"):
        z, x, y = tuple(map(int, tile_name.split("_")))
        if tile_name not in image_dict:
            group_coord = get_parent_tile(x, y)
            image_path = os.path.join(
                image_dir, f"14_{group_coord[0]}_{group_coord[1]}.webp"
            )
            temp_dict = load_images_from_path(image_path)
            for (x, y), v in temp_dict.items():
                # Convert tensor to NumPy array and scale
                np_img = (v.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                # If the image has 3 channels (RGB), convert to RGBA by adding an alpha channel
                if np_img.shape[2] == 3:
                    alpha_channel = np.full(
                        (np_img.shape[0], np_img.shape[1], 1), 255, dtype=np.uint8
                    )
                    np_img = np.concatenate((np_img, alpha_channel), axis=2)

                # Encode the image to PNG format
                success, buffer = cv2.imencode(".png", np_img)
                if not success:
                    raise ValueError("Failed to encode image to PNG format.")

                # Base64 encode the PNG bytes
                base64_str = base64.b64encode(buffer).decode("utf-8")

                # Create the data URI
                data_uri = f"data:image/png;base64,{base64_str}"
                image_dict[f"16_{x}_{y}"] = data_uri
    return image_dict


def init_embeddings(args):
    # Load embeddings

    cls_h5_path = args.embedding_dir + "/cls.h5"  # Update with actual path
    feat_h5_path = None  # Update with actual path or None
    image_dir = args.image_dir  # Update with actual path

    embeddings_dict, embedding_dim = load_embeddings(cls_h5_path, feat_h5_path)
    group_names, all_embeddings, reduced = compute_embeddings(embeddings_dict)
    image_dict = load_images(group_names, image_dir, None)

    return {
        "group_names": group_names,
        "all_embeddings": all_embeddings.tolist(),
        "reduced": reduced.tolist(),
        "image_dict": image_dict,
    }


parser = argparse.ArgumentParser(
    description="Perform kNN search on HDF5 embeddings with image visualization using UMAP."
)
parser.add_argument(
    "-e",
    "--embedding-dir",
    type=str,
    required=True,
    help="Path to dir with cls.h5 file",
)
parser.add_argument(
    "-img",
    "--image-dir",
    type=str,
    default=None,
    help="Directory containing group images (.webp files)",
)
args = parser.parse_args()
data = init_embeddings(args)
knn_model = NearestNeighbors(n_neighbors=6, algorithm="auto").fit(
    data["all_embeddings"]
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "kNN Interactive Visualization"


@app.callback(
    Output("embedding-graph", "figure"),
    Output("info", "children"),
    Input("embedding-graph", "clickData"),
    State("embedding-data", "data"),
)
def update_graph(clickData, data):
    print("update_graph")
    start = time.time()
    if not data:
        raise dash.exceptions.PreventUpdate

    group_names = data["group_names"]
    all_embeddings = np.array(data["all_embeddings"])
    reduced = np.array(data["reduced"])
    image_dict = data["image_dict"]

    # Initialize figure
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=reduced[:, 0],
            y=reduced[:, 1],
            mode="markers",
            marker=dict(
                size=10,
                color="lightgray",
                opacity=0.7,
                line=dict(width=1, color="white"),
            ),
            text=group_names,
            hoverinfo="text",
            name="All Groups",
        )
    )

    selected_group = None
    neighbors = []
    images_html = ""

    if clickData:
        clicked_point = clickData["points"][0]
        selected_group = clicked_point["text"]
        selected_index = group_names.index(selected_group)
        selected_embedding = all_embeddings[selected_index].reshape(1, -1)

        # Perform kNN search
        distances, indices = knn_model.kneighbors(selected_embedding)
        distances = distances[0][1:]  # Exclude self
        indices = indices[0][1:]
        neighbors = [group_names[idx] for idx in indices]

        # Highlight selected group
        fig.add_trace(
            go.Scatter(
                x=[reduced[selected_index, 0]],
                y=[reduced[selected_index, 1]],
                mode="markers",
                marker=dict(
                    size=15,
                    color="red",
                    opacity=1.0,
                    line=dict(width=2, color="darkred"),
                ),
                text=[selected_group],
                hoverinfo="text",
                name="Selected Group",
            )
        )

        # Highlight neighbors
        fig.add_trace(
            go.Scatter(
                x=reduced[indices, 0],
                y=reduced[indices, 1],
                mode="markers",
                marker=dict(
                    size=12,
                    color="blue",
                    opacity=1.0,
                    line=dict(width=1, color="darkblue"),
                ),
                text=neighbors,
                hoverinfo="text",
                name="kNN",
            )
        )

        # Prepare images HTML
        selected_image = image_dict.get(selected_group)

        def get_dist(name1, name2):
            coords1 = list(map(int, name1.split("_")))
            coords2 = list(map(int, name2.split("_")))
            return np.linalg.norm(np.array(coords1) - np.array(coords2))

        neighbor_images = {n: image_dict.get(n) for n in neighbors}
        images_html = html.Div(
            [
                html.H3(f"Selected Group: {selected_group}"),
                html.Img(
                    src=selected_image, style={"width": "150px", "height": "150px"}
                ),
                html.H4("Nearest Neighbors:"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Img(
                                    src=img,
                                    style={
                                        "width": "100px",
                                        "height": "100px",
                                        "margin": "5px",
                                    },
                                ),
                                html.P(
                                    name,
                                    style={"textAlign": "center", "padding": "4px"},
                                ),
                                html.P(f"emb dist: {distances[i]:.1f}"),
                                html.P(
                                    f"dist: {get_dist(name, selected_group):.0f} tiles",
                                    style={"textAlign": "center", "padding": "4px"},
                                ),
                            ]
                        )
                        for i, (name, img) in enumerate(neighbor_images.items())
                        if img
                    ],
                    style={"display": "flex", "justifyContent": "center"},
                ),
            ]
        )
    else:
        images_html = "Click on a point to see its nearest neighbors."

    # Update layout
    fig.update_layout(
        title="Embeddings Visualization with kNN",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        hovermode="closest",
        legend=dict(x=0.7, y=0.95),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    print(f"update_graph took {time.time() - start:.2f} seconds")
    return fig, images_html


if __name__ == "__main__":
    # Initialize the Dash app

    # Define the app layout
    app.layout = html.Div(
        [
            html.H1("kNN Interactive Visualization", style={"textAlign": "center"}),
            # Hidden div to store data
            dcc.Store(id="embedding-data", data=data),
            # Visualization area
            dcc.Graph(
                id="embedding-graph",
                config={"displayModeBar": True},
                style={"height": "80vh"},
            ),
            # Information area
            html.Div(id="info", style={"textAlign": "center", "marginTop": "20px"}),
        ]
    )
    app.run_server(debug=True)
