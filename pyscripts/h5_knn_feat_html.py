import argparse
import base64
import dataclasses
import os
import warnings
from collections import defaultdict
from io import BytesIO
import rootutils

import processed_tile_group_pb2
from h5_knn_html import load_image, get_parent_tile

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import h5py
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from PIL import Image, ImageDraw
from src.modules.embedding_lookup import EmbeddingLookup
from src.modules.tag_encoder import TagIndexer
from src.modules.tokenizer import Modality, compute_normalized_bboxes
from h5_read import FakeTileBatch

# Initialize tag indexer if needed
tag_indexer = TagIndexer("data/tiles/embeddings.pkl")


def pil_to_base64(pil_img: Image.Image):
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


from PIL import Image
from typing import List


from PIL import Image


@dataclasses.dataclass
class Point:
    lat: float
    lon: float

    def is_close(self, other):
        return abs(self.lat - other.lat) < 1e-6 and abs(self.lon - other.lon) < 1e-6


@dataclasses.dataclass
class CustomGeometry:
    points: List
    is_closed: bool


pbf_tile_cache = {}


def get_processed_pbf_tile(name):
    if name not in pbf_tile_cache:
        tile_coords = list(map(int, name.split("_")))
        px, py = get_parent_tile(tile_coords[1], tile_coords[2])
        parent_name = f"14_{px}_{py}"
        tile_group = processed_tile_group_pb2.TileGroup()
        with open(
            f"data/tiles/huge/tasks/pretraining/test/{parent_name}.pbf", "rb"
        ) as f:
            tile_group.ParseFromString(f.read())
        for tile in tile_group.tiles:
            pbf_tile_cache[f"16_{tile.x}_{tile.y}"] = tile
    return pbf_tile_cache[name]


def extract_feature_geometry(tile, feature_id) -> CustomGeometry:
    """
    Extracts the geometry for a specific feature from the tile.

    :param tile: An object containing nodes, inter_edges, intra_edges, and node_to_feature.
    :param feature_id: The ID of the feature to extract.
    :return: A CustomGeometry object for the given feature ID.
    """
    node_to_feature = torch.tensor(tile.node_to_feature, dtype=torch.long)
    nodes = torch.tensor(tile.nodes, dtype=torch.float32).reshape(-1, 2)
    # Extract intra_edges: reshape to [2, E_intra]
    edges = torch.tensor(tile.intra_edges, dtype=torch.long)
    intra_edges = torch.stack((edges[::2], edges[1::2]))

    # Extract inter_edges: reshape to [2, E_inter]
    edges = torch.tensor(tile.inter_edges, dtype=torch.long)
    inter_edges = torch.stack((edges[::2], edges[1::2]))

    # Combine inter and intra edges
    all_edges = torch.cat(
        [
            inter_edges,
            # intra_edges
        ],
        dim=1,
    )

    # Convert tensors to numpy arrays for ease of use
    node_to_feat_np = (
        node_to_feature.cpu().numpy()
        if node_to_feature.is_cuda
        else node_to_feature.numpy()
    )
    nodes_np = nodes.cpu().numpy() if nodes.is_cuda else nodes.numpy()
    edges_np = all_edges.cpu().numpy() if all_edges.is_cuda else all_edges.numpy()

    collected_points = []

    # Iterate through each edge
    for edge in edges_np.T:
        u, v = int(edge[0]), int(edge[1])
        # Check if both endpoints belong to the specified feature
        if node_to_feat_np[u] == feature_id and node_to_feat_np[v] == feature_id:
            # Extract coordinates for the two nodes
            p1_coords = nodes_np[u].tolist()  # [lat, lon]
            p2_coords = nodes_np[v].tolist()  # [lat, lon]

            # Create two Point objects for the edge
            point1 = Point(lat=p1_coords[0], lon=p1_coords[1])
            point2 = Point(lat=p2_coords[0], lon=p2_coords[1])

            # Add points if they are not already collected
            if point1 not in collected_points:
                collected_points.append(point1)
            if point2 not in collected_points:
                collected_points.append(point2)

    # Determine if the geometry is closed: if first and last points are the same
    is_closed = tile.features[feature_id].is_polygon

    return CustomGeometry(points=collected_points, is_closed=is_closed)


from PIL import Image


def create_feature_overlay_image(meta, margin_pixels=20, target_size=224):
    tile = meta["tile"]
    min_box = meta["min_box"]
    mod = meta["mod"]

    # Early return for image modality
    if mod == Modality.IMG:
        return draw_bboxes_on_image(tile, min_box)

    feat_id = meta["in_mod_id"]
    pbf_file = get_processed_pbf_tile(tile.name())
    geo = extract_feature_geometry(pbf_file, feat_id)

    # Load the original image as RGBA
    np_img = load_image(tile.name(), "data/tiles/huge/images", return_np=True)
    pil_img = Image.fromarray(np_img).convert("RGBA")

    # Compute the pixel coordinates for the min_box boundaries
    box_x = min_box[:, 0] * pil_img.width
    box_y = (1 - min_box[:, 1]) * pil_img.height

    # Determine the bounding box of the min_box in pixel space
    min_x, max_x = box_x.min().item(), box_x.max().item()
    min_y, max_y = box_y.min().item(), box_y.max().item()

    # Expand the bounding box by the specified margin
    expanded_left = max(0, min_x - margin_pixels)
    expanded_top = max(0, min_y - margin_pixels)
    expanded_right = min(pil_img.width, max_x + margin_pixels)
    expanded_bottom = min(pil_img.height, max_y + margin_pixels)

    # Ensure the crop region is square by adjusting width/height if necessary
    box_width = expanded_right - expanded_left
    box_height = expanded_bottom - expanded_top

    if box_width > box_height:
        diff = box_width - box_height
        expanded_top = max(0, expanded_top - diff // 2)
        expanded_bottom = min(pil_img.height, expanded_bottom + diff // 2)
    elif box_height > box_width:
        diff = box_height - box_width
        expanded_left = max(0, expanded_left - diff // 2)
        expanded_right = min(pil_img.width, expanded_right + diff // 2)

    # Crop the image to the computed square region
    cropped_img = pil_img.crop(
        (expanded_left, expanded_top, expanded_right, expanded_bottom)
    )

    # Resize the cropped image to 224x224
    resized_img = cropped_img.resize((target_size, target_size))

    # Determine scaling factors based on the crop and resize
    crop_width = expanded_right - expanded_left
    crop_height = expanded_bottom - expanded_top

    # Transform original geometry coordinates to the resized image coordinate system
    new_x = [
        ((c.lon * pil_img.width - expanded_left) / crop_width) * target_size
        for c in geo.points
    ]
    new_y = [
        (((1 - c.lat) * pil_img.height - expanded_top) / crop_height) * target_size
        for c in geo.points
    ]

    # Prepare to draw on the resized image
    draw = ImageDraw.Draw(resized_img)

    # Draw geometry based on its type
    if geo.is_closed and len(geo.points) > 2:
        xy = list(zip(new_x, new_y))
        # Create an overlay for transparency
        overlay = Image.new("RGBA", resized_img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        fill_color = (0, 0, 255, 128)  # semi-transparent blue
        outline_color = (0, 0, 255, 255)  # solid blue
        overlay_draw.polygon(xy, outline=outline_color, fill=fill_color)
        # Composite the overlay with the resized image
        resized_img = Image.alpha_composite(resized_img, overlay)
    else:
        xy = list(zip(new_x, new_y))
        if len(geo.points) <= 1:
            # If only a single point, draw a small ellipse
            center_x = new_x[0] if new_x else target_size / 2
            center_y = new_y[0] if new_y else target_size / 2
            radius = 5
            draw.ellipse(
                (
                    center_x - radius,
                    center_y - radius,
                    center_x + radius,
                    center_y + radius,
                ),
                fill="red",
            )
        else:
            # For open geometries, draw a line connecting the points
            draw.line(xy, fill="orange", width=5)

    return resized_img


def draw_bboxes_on_image(tile, min_box):
    # Load tile image (assuming similar logic as in the second script)
    np_img = load_image(tile.name(), "data/tiles/huge/images", return_np=True)
    pil_img = Image.fromarray(np_img)
    draw = ImageDraw.Draw(pil_img)
    #   coords = list((min_box * np.array([pil_img.width, pil_img.height])).reshape(-1))
    # /Users/theo/Documents/courses/current/master-thesis/geojepa/pyscripts/h5_knn_feat_html.py:41: DeprecationWarning: __array_wrap__ must accept context and return_scalar arguments (positionally) in the future. (Deprecated NumPy 2.0)
    # disable warning
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # reorder lat and lon
    x = min_box[:, 0] * pil_img.width
    y = (1 - min_box[:, 1]) * pil_img.height
    xy = []
    for i in range(4):
        xy.append(x[i])
        xy.append(y[i])
    draw.polygon(xy, outline="red", width=3)
    return pil_img


def get_in_mod_index(tile_mods, token_id):
    mod = tile_mods[token_id].item()
    first_mod_index = list(tile_mods).index(mod)
    return token_id - first_mod_index


def get_min_box_for_feature(tile, mods, token_id):
    # Use logic from second script to compute min_box based on modality
    # Adapted to work in this context; ensure proper imports and definitions

    mod = mods[token_id].item()
    in_mod_token_idx = get_in_mod_index(mods, token_id)

    if mod == Modality.OSM:
        # print(".......")
        # print(tile.min_boxes.shape)
        # print(mods.shape, token_id, in_mod_token_idx)
        return tile.min_boxes[in_mod_token_idx]
    elif mod == Modality.IMG:
        if in_mod_token_idx == 0:
            return torch.tensor(((0, 0), (1, 0), (1, 1), (0, 1))).reshape(4, 2)
        box = compute_normalized_bboxes(1, 14, device="cpu")[0][in_mod_token_idx - 1]
        return box.reshape(4, 2)
    else:
        return None
        pass
        raise ValueError(f"Unsupported modality [{Modality(mod).name}]")


from src.data.components.tile_dataset import TileDataset


class TileReader:
    def __init__(self, task="pretraining", split="test", size="huge", cheat=False):
        self.dataset = TileDataset(
            f"data/tiles/{size}/tasks/{task}",
            split,
            cheat=cheat,
            use_image_transforms=False,
            load_images=False,
            add_original_image=False,
        )
        self.dataset.__init_worker__()
        self.tile_ids = None

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def get_tile_by_name(self, name):
        if self.tile_ids is None:
            self.tile_ids = {}
            for key in tqdm(self.dataset.file.keys(), desc="Loading tile IDs"):
                group = self.dataset.file[key]
                tile_coord = torch.tensor(group["tile_coord"][:]).tolist()
                tile_name = f"{tile_coord[0]}_{tile_coord[1]}_{tile_coord[2]}"
                self.tile_ids[tile_name] = key
        idx = self.tile_ids[name]
        tile = self.dataset.__getitem__(idx)
        assert tile.name() == name
        return tile

    def __len__(self):
        return self.dataset.__len__()


def init_feature_embeddings(args, tile_reader):
    embedding_dir = args.embedding_dir
    image_dir = args.image_dir

    # Initialize embedding lookup for features
    embedding_model = EmbeddingLookup(embedding_dir, cls_only=False, mods=True)

    # Containers to store feature-level information
    feature_embeddings = []
    feature_ids = []  # Unique ID for each feature (e.g., tileID_featureIndex)
    all_mods = []
    id_to_name_map = {}
    feature_metadata = {}  # Store metadata like tile reference, token_id, modality, min_box, etc.

    # For simplicity, assume a list of tile names is available from identifiers.h5 (or similar source)
    cls_h5_path = os.path.join(embedding_dir, "cls.h5")
    id_path = cls_h5_path.replace("cls.h5", "identifiers.h5")
    with h5py.File(id_path, "r") as f:
        tile_ids = f["identifiers"][:]
    if isinstance(tile_ids[0], bytes):
        tile_ids = [id.decode("utf-8") for id in tile_ids]
    else:
        tile_ids = [str(id) for id in tile_ids]

    np.random.seed(42)
    np.random.shuffle(tile_ids)
    if args.tiles:
        tile_ids = tile_ids[: args.tiles]

    print(f"Processing {len(tile_ids)} tiles for feature extraction...")

    for tile_name in tqdm(tile_ids, desc="Extracting Tile Features"):
        # Load tile using FakeTileBatch for the given tile_name
        tile_batch = FakeTileBatch(tile_name)
        # Get embeddings: cls, features, modalities
        _, feats, mods = embedding_model(tile_batch)
        # Suppose you have access to the tile object for additional info (min_boxes etc.)
        # For demonstration, assume a function get_tile_by_name exists:
        tile = tile_reader.get_tile_by_name(
            tile_name
        )  # You need to implement this based on your dataset

        feats = feats[0]  # Extract from batch dimension
        mods = mods[0]
        feats = feats[mods != Modality.PAD]
        mods = mods[mods != Modality.PAD]

        for token_id, token in enumerate(feats):
            feature_id = f"{tile_name}_feat_{token_id}"
            feature_embeddings.append(token.detach().cpu().numpy())
            feature_ids.append(feature_id)
            id_to_name_map[len(feature_embeddings) - 1] = feature_id
            all_mods.append(mods[token_id].item())
            # Store metadata for later use in HTML rendering
            feature_metadata[feature_id] = {
                "tile": tile,
                "embedding_id": len(feature_embeddings) - 1,
                "token_id": token_id,
                "mod": mods[token_id].item(),
                "in_mod_id": get_in_mod_index(mods, token_id),
                "min_box": get_min_box_for_feature(tile, mods, token_id),
            }

    # shuffle feature_ids, embeddings and metadata in the same wayt
    # perm = np.random.permutation(len(feature_ids))
    # feature_ids = [feature_ids[i] for i in perm]
    # feature_embeddings = [feature_embeddings[i] for i in perm]

    feature_embeddings = np.vstack(feature_embeddings)
    all_mods = np.array(all_mods)
    print(f"Total features loaded: {len(feature_ids)}")

    # Fit kNN model on feature embeddings
    n = args.n_neighbours
    if args.hide_same_tile:
        n = 50
    knn_model = NearestNeighbors(n_neighbors=n + 1, algorithm="auto").fit(
        feature_embeddings
    )

    return {
        "feature_ids": feature_ids,
        "id_to_name_map": id_to_name_map,
        "all_mods": all_mods,
        "embeddings": feature_embeddings,
        "knn_model": knn_model,
        "metadata": feature_metadata,
        "image_dir": image_dir,
    }


def get_feature_tags(in_mod_id, tile, top=5):
    global tag_indexer
    if tag_indexer is None:
        tag_indexer = TagIndexer("data/tiles/embeddings.pkl")
    tag_count = defaultdict(int)
    tags = tile.tags[in_mod_id]
    for tag_id in tags:
        if tag_id != 0:
            k, v = tag_indexer.index_to_tag(tag_id)
            if k != "PAD":
                tag_count[k, v] += 1

    # get top5
    tags = dict(
        sorted(tag_count.items(), key=lambda item: len(item[0][1]), reverse=True)
    )
    display = [f"[{k}={v}]" for (k, v), count in tags.items()]
    if len(display) > top:
        display = display[:top]
    not_covered = len(tag_count) - len(display)
    if not_covered > 0:
        display.append(f"and {not_covered} more..")
    return display


def display_image(img, border=False):
    img_style = "border-radius:5px;"
    img_h = 224
    img_w = 224
    if border:
        img_style += "border-style: solid; border-color: red; border-width: 2px;"
        img_h = 220
        img_w = 220

    return f"""
        <td style="padding: 5px;">
            <img src="{img}" width="{img_w}" height="{img_h}" style="{img_style}"/>
        </td>
    """


def display_tags(query_meta, meta):
    dist = 0
    if query_meta["mod"] == Modality.OSM:
        dist = fmt_dist(get_dist(query_meta, meta))
    mod = Modality(meta["mod"])
    in_mod_id = meta["in_mod_id"]
    tags = []
    if mod == Modality.OSM:
        tags = get_feature_tags(in_mod_id, meta["tile"])
    return f"""
            <td style="">
                <div style="padding-left: 20px;overflow-x:hidden;display: flex; justify-items: self-start; justify-self: auto; flex-direction: column; row-gap: 2px; align-self: start;">
                    <div>
                        {"".join([f"<strong>{tag}</strong><br/>" for tag in tags])}
                    </div>
                    <div style="margin-top:10px;">
                        <span><strong>Dist:</strong> {dist}</span>
                    </div>
                    <div>
                        <span><strong>Same tile:</strong> {"yes" if query_meta["tile"].name() == meta["tile"].name() else "no"}</span>
                    </div>
                </div>
            </td>
    """


def fmt_dist(dist):
    # m or km
    if dist < 1000:
        return f"{dist:.0f}m"
    return f"{dist / 1000:.1f}km"


def get_dist(feat1meta, feat2meta):
    tile1 = feat1meta["tile"]
    tile2 = feat2meta["tile"]
    f1box = feat1meta["min_box"]
    f2box = feat2meta["min_box"]
    if f1box is None or f2box is None:
        return -1
    f1center = f1box.float().mean(axis=0) * 300.0
    f2center = f2box.float().mean(axis=0) * 300.0
    coords1 = list(map(int, tile1.name().split("_")))
    coords2 = list(map(int, tile2.name().split("_")))
    return np.linalg.norm(np.array(coords1) - np.array(coords2)) * 300 + np.linalg.norm(
        f1center - f2center
    )


def generate_feature_table_html(file_name, data, args):
    feature_ids = data["feature_ids"]
    embeddings = data["embeddings"]
    knn_model = data["knn_model"]
    metadata = data["metadata"]
    all_mods = data["all_mods"]
    image_dir = data["image_dir"]
    id_to_name_map = data["id_to_name_map"]

    n = args.num_features if hasattr(args, "num_features") else 50
    if n > len(feature_ids):
        n = len(feature_ids)

    # Select n random feature queries (shuffled earlier or sample subset)
    query_feature_ids = []
    perm = np.random.permutation(len(feature_ids))
    feature_ids = [feature_ids[i] for i in perm]
    # feature_embeddings = [feature_embeddings[i] for i in perm]
    for id in feature_ids:
        if len(query_feature_ids) == n:
            break
        if metadata[id]["mod"] == Modality.OSM:
            query_feature_ids.append(id)

    table_rows = []

    for fid in tqdm(query_feature_ids, desc="Rendering Features"):
        # Load query metadata
        meta = metadata[fid]

        # Use the tile image with bounding box for the feature
        # query_img = draw_bboxes_on_image(tile, min_box)
        query_img = create_feature_overlay_image(meta)
        query_img_base64 = pil_to_base64(query_img)

        # Find nearest neighbors
        embedding_id = meta["embedding_id"]
        embedding = embeddings[embedding_id].reshape(1, -1)
        distances, indices = knn_model.kneighbors(embedding)

        # Exclude self in neighbors
        neighbor_indices = indices[0]
        neighbor_ids = [id_to_name_map[i] for i in neighbor_indices]
        # remove self
        neighbor_ids = [id for id in neighbor_ids if id != fid]
        if args.hide_same_tile:
            neighbor_ids = [
                id
                for id in neighbor_ids
                if metadata[id]["tile"].name() != meta["tile"].name()
            ]
            if len(neighbor_ids) > args.n_neighbours:
                neighbor_ids = neighbor_ids[: args.n_neighbours]

        neighbor_imgs = []

        for feat_id in neighbor_ids:
            nb_meta = metadata[feat_id]
            nb_img = create_feature_overlay_image(nb_meta)
            nb_img_base64 = pil_to_base64(nb_img)
            neighbor_imgs.append(nb_img_base64)

        # Construct HTML row for this feature and its neighbors
        row_html = f"""
        <tr>
            {display_image(query_img_base64, border=True)}
            {"".join([display_image(img, border=False) for img in neighbor_imgs])}
        </tr>
        <tr>
            {display_tags(meta, meta)}
            {"".join([display_tags(meta, metadata[id]) for id in neighbor_ids])}
        </tr>
        <tr>
            {
            "".join(
                [
                    f"<td style='padding: 5px;'><p>{id}</p></td>"
                    for id in [fid] + neighbor_ids
                ]
            )
        }
        </tr>
        """
        table_rows.append(row_html)

    table_html = f"""
    <table cellspacing="0" cellpadding="0" style="border-collapse:collapse;border:none;">
      {"".join(table_rows)}
    </table>
    """
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{file_name}</title>
    </head>
    <body>
        <h1>Displaying {n} Random Features</h1>
        <h2>{file_name}</h2>
        {table_html}
    </body>
    </html>
    """
    return full_html


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature-level kNN search with image visualization, generating an HTML file."
    )
    parser.add_argument(
        "-e",
        "--embedding-dir",
        type=str,
        required=True,
        help="Path to the embeddings directory",
    )
    parser.add_argument(
        "-img",
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing tile images (.webp files)",
    )
    parser.add_argument(
        "-n",
        "--num_features",
        type=int,
        default=50,
        help="Number of random features to display in the HTML output",
    )
    parser.add_argument(
        "-k",
        "--n_neighbours",
        type=int,
        default=4,
        help="Number of nearest neighbours to display",
    )
    parser.add_argument(
        "--tiles", type=int, default=1000, help="Limit number of tiles to process"
    )
    parser.add_argument(
        "--hide-same-tile", action="store_true", help="Remove same tile features"
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Custom name, otherwise derived from embedding directory",
    )
    args = parser.parse_args()

    feature_dir_name = "-".join(args.embedding_dir.split("/")[-2:])
    tile_reader = TileReader()
    data = init_feature_embeddings(args, tile_reader)
    if args.save_name:
        feature_dir_name = args.save_name
    html_output = generate_feature_table_html(feature_dir_name, data, args)

    output_dir = "pyscripts/out/knn_features/"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_dir}/{feature_dir_name}.html"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_output)

    print(f"HTML file successfully generated: {output_filename}")
