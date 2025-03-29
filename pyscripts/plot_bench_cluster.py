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
from matplotlib import pyplot as plt
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



from src.modules.tag_encoder import TagIndexer


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

    # Ensure tile_ids are strings (handles both bytes and numeric cases)
    if isinstance(tile_ids[0], bytes):
        tile_ids = [id.decode("utf-8") for id in tile_ids]
    else:
        tile_ids = [str(id) for id in tile_ids]

    # Shuffle and optionally limit the tile list
    np.random.seed(42)
    np.random.shuffle(tile_ids)
    if args.tiles:
        tile_ids = tile_ids[: args.tiles]

    print(f"Processing {len(tile_ids)} tiles for feature extraction...")

    for tile_name in tqdm(tile_ids, desc="Extracting Tile Features"):
        # Load tile using FakeTileBatch for the given tile_name
        tile = tile_reader.get_tile_by_name(tile_name)
        for osm_tags in tile.tags:
            # Check if any (k, v) contains "bench"
            has_bench = False
            for tag_id in osm_tags:
                if tag_id != 0:  # skip padding tag
                    k, v = tag_indexer.index_to_tag(tag_id)
                    # Case-insensitive substring check
                    if args.search in ":".join([k.lower(), v.lower()]):
                        has_bench = True
                        break
            if not has_bench:
                continue
        # Retrieve the tile object for tags / boxes
        tile_batch = FakeTileBatch(tile_name)
        # Get embeddings: cls, features, modalities
        _, feats, mods = embedding_model(tile_batch)

        # Remove padding tokens
        feats = feats[0][mods[0] != Modality.PAD]
        mods = mods[0][mods[0] != Modality.PAD]

        # Iterate through features for the tile
        for token_id, token in enumerate(feats):
            mod_val = mods[token_id].item()

            # We only check "bench" for OSM tokens
            if mod_val == Modality.OSM:
                # in_mod_id used to index into tile.tags, etc.
                in_mod_id = get_in_mod_index(mods, token_id)
                osm_tags = tile.tags[in_mod_id]

                # Check if any (k, v) contains "bench"
                has_bench = False
                for tag_id in osm_tags:
                    if tag_id != 0:  # skip padding tag
                        k, v = tag_indexer.index_to_tag(tag_id)
                        # Case-insensitive substring check
                        if args.search in ":".join([k.lower(), v.lower()]):
                            has_bench = True
                            break
                if not has_bench:
                    # Skip this feature if it doesn't contain bench
                    continue
            else:
                # Skip any non-OSM feature if you only want benches
                continue

            # If we reach here:
            # -> OSM feature that has bench in its tags -> store it
            feature_id = f"{tile_name}_feat_{token_id}"
            feature_embeddings.append(token.detach().cpu().numpy())
            feature_ids.append(feature_id)
            id_to_name_map[len(feature_embeddings) - 1] = feature_id
            all_mods.append(mod_val)

            # Store metadata for later use
            in_mod_id = get_in_mod_index(mods, token_id)
            feature_metadata[feature_id] = {
                "tile": tile,
                "embedding_id": len(feature_embeddings) - 1,
                "token_id": token_id,
                "mod": mod_val,
                "in_mod_id": in_mod_id,
                "min_box": get_min_box_for_feature(tile, mods, token_id),
            }

    if len(feature_embeddings) == 0:
        print(f"No features with '{args.search}' found!")
        return {
            "feature_ids": [],
            "id_to_name_map": {},
            "all_mods": np.array([]),
            "embeddings": np.array([]),
            "metadata": {},
            "image_dir": image_dir,
        }

    # Convert to numpy arrays
    feature_embeddings = np.vstack(feature_embeddings)
    all_mods = np.array(all_mods)

    print(f"Total {args.search} features loaded: {len(feature_ids)}")

    return {
        "feature_ids": feature_ids,
        "id_to_name_map": id_to_name_map,
        "all_mods": all_mods,
        "embeddings": feature_embeddings,
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



# If not already installed:
from pacmap import PaCMAP
from sklearn.cluster import KMeans


def extract_bench_features(keyword, data):
    """
    Args:
        data (dict): The dictionary from init_feature_embeddings with:
            - 'feature_ids'
            - 'id_to_name_map'
            - 'all_mods'
            - 'embeddings'
            - 'metadata'
            - 'image_dir'
    """
    bench_embeddings = []
    bench_ids = []

    metadata = data["metadata"]
    all_embeddings = data["embeddings"]
    feature_ids = data["feature_ids"]

    for idx, feat_id in tqdm(enumerate(feature_ids), "Filtering embeddings"):
        meta = metadata[feat_id]
        mod = meta["mod"]
        in_mod_id = meta["in_mod_id"]

        # We only look for OSM features (since tags are relevant to OSM)
        if mod == Modality.OSM:
            tile = meta["tile"]
            # Get the tags for this feature (in_mod_id)
            # Reusing the same logic from get_feature_tags
            tags = tile.tags[in_mod_id]
            # Check if any k=v pair contains "bench"
            for tag_id in tags:
                if tag_id == 0:
                    continue
                k, v = tag_indexer.index_to_tag(tag_id)
                if keyword in ":".join([k.lower(), v.lower()]):
                    bench_embeddings.append(all_embeddings[idx])
                    bench_ids.append(feat_id)
                    break  # No need to check other tags for this feature

    bench_embeddings = np.array(bench_embeddings)
    return bench_embeddings, bench_ids



# pip install pacmap

# All your previous imports and definitions here ...
# e.g. TileReader, init_feature_embeddings, create_feature_overlay_image,
# pil_to_base64, get_feature_tags, etc.


def render_bench_html_cell(bench_id, data, img_size=128):
    """
    Given a feature ID (bench_id) and the full data dictionary,
    returns an HTML <td> cell containing the base64 image overlay
    and some metadata (like tags).
    """
    meta = data["metadata"][bench_id]

    # Create overlay image (geometry or bounding box)
    overlay_img = create_feature_overlay_image(meta)
    b64_img = pil_to_base64(overlay_img)

    # Fetch tags
    tags_html = ""
    if meta["mod"] == Modality.OSM:
        # Show top 5 tags by default
        in_mod_id = meta["in_mod_id"]
        bench_tags = get_feature_tags(in_mod_id, meta["tile"], top=5)
        if bench_tags:
            tags_html = "<br/>".join([f"<strong>{t}</strong>" for t in bench_tags])

    cell_html = f"""
      <div style="text-align:center;">
        <img src="{b64_img}" width="{img_size}" height="{img_size}" style="border-radius:4px;"/>
        <div style="margin-top:5px;">{tags_html}</div>
      </div>
    """
    return cell_html


def generate_clustered_html(
    bench_ids, pacmap_X, kmeans_labels, data, n_clusters=5, max_per_cluster=10
):
    """
    Create an HTML string that displays the benches grouped by cluster.
    Each cluster is one row, and each bench in that cluster is shown as a column.

    Args:
        bench_ids (list[str]): The feature IDs of the benches (same order as pacmap_X).
        pacmap_X (np.array): The 2D PaCMAP embedding of shape [N, 2].
        kmeans_labels (np.array): The cluster labels for each of the N benches.
        data (dict): The dictionary from init_feature_embeddings (containing metadata, etc.).
        n_clusters (int): Number of clusters.
        max_per_cluster (int): Maximum benches to display per cluster.

    Returns:
        str: The complete HTML string.
    """
    html = []
    html.append("<html><head>")
    html.append("<meta charset='utf-8'>")
    html.append("<title>Bench Clusters</title>")
    html.append("</head><body>")
    html.append("<h1>PaCMAP + KMeans Clusters for 'Bench' Features</h1>")
    html.append("<table style='border-spacing: 10px 10px; width:100%;'>")

    for c in range(n_clusters):
        # Get all indices belonging to cluster c
        cluster_indices = np.where(kmeans_labels == c)[0]
        if len(cluster_indices) == 0:
            continue

        # Optionally shuffle or select top max_per_cluster
        # For a stable output, you can skip shuffling:
        # np.random.shuffle(cluster_indices)
        # cluster_indices = cluster_indices[:max_per_cluster]

        # Start a new row for this cluster
        row_html = []
        row_html.append("<tr style='vertical-align:top;'>")

        # The first cell can be a header with the cluster ID
        row_html.append(f"<td><h2>Cluster {c}</h2></td>")

        # Each subsequent cell is a bench in this cluster
        count = 0
        first_tile = None
        for idx in cluster_indices:
            bench_id = bench_ids[idx]
            tile_name = data["metadata"][bench_id]["tile"].name()
            if not first_tile:
                first_tile = tile_name
            if first_tile != tile_name:
                count += 1
                cell_html = render_bench_html_cell(bench_id, data, img_size=128)
                row_html.append(f"<td>{cell_html}</td>")

        row_html.append("</tr>")
        html.append("".join(row_html))

    html.append("</table></body></html>")
    return "".join(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract 'bench' features, create PaCMAP scatter, and generate HTML clusters."
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
        "--tiles", type=int, default=1000, help="Limit number of tiles to process"
    )
    parser.add_argument(
        "--hide-same-tile",
        action="store_true",
        help="Remove same tile features (unused in bench-only scenario).",
    )
    parser.add_argument(
        "-k",
        "--n_neighbours",
        type=int,
        default=4,
        help="Number of nearest neighbours (not directly used here).",
    )
    parser.add_argument(
        "-s",
        "--search",
        type=str,
        default="bench",
        help="Keyword to filter in tags, default is 'bench'.",
    )
    parser.add_argument(
        "--max-per-cluster",
        type=int,
        default=10,
        help="Maximum features to display per cluster in the HTML table.",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=5, help="Number of k-means clusters."
    )
    args = parser.parse_args()

    # 1) Initialize a TileReader
    tile_reader = TileReader()

    # 2) Extract bench-only embeddings via init_feature_embeddings
    data = init_feature_embeddings(
        args, tile_reader
    )  # <--- already filters for 'bench'

    # If no features found, exit
    if len(data["embeddings"]) == 0:
        print("No features with 'bench' found. Exiting.")
        exit(0)

    # 3) Create a PaCMAP embedding
    embeddings = data["embeddings"]
    bench_ids = data["feature_ids"]

    reducer = PaCMAP(n_components=2, random_state=42)
    print(f"Running PaCMAP on {embeddings.shape[0]} bench features...")
    pacmap_X = reducer.fit_transform(embeddings, init="pca")

    # 4) K-means clustering
    n_clusters = args.n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(pacmap_X)

    # 5) (Optional) Visual scatter
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        pacmap_X[:, 0], pacmap_X[:, 1], c=kmeans_labels, cmap="tab10", alpha=0.7
    )
    plt.title(f"PaCMAP of '{args.search}' Features with K-Means Clusters")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    plt.show()

    # 6) Generate HTML for clusters, one row per cluster
    html_str = generate_clustered_html(
        bench_ids,
        pacmap_X,
        kmeans_labels,
        data,
        n_clusters=n_clusters,
        max_per_cluster=args.max_per_cluster,
    )

    # 7) Write HTML file
    output_dir = "pyscripts/out/knn_features/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"{args.search}_{args.n_clusters}_clusters.html"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"Done! HTML of bench clusters saved to: {output_path}")
