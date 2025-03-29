import argparse

import rootutils

from h5_read import FakeTileBatch
from src.modules.embedding_lookup import EmbeddingLookup

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from collections import defaultdict
from io import BytesIO
from os import makedirs

import cv2
import h5py
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

import os
import base64

from src.data.components.raw_tile_dataset import load_images_from_path
from src.modules.tag_encoder import TagIndexer
from src.utils.tile_utils import tile_zxy_to_latlon_bbox
import unprocessed_tile_group_pb2 as unprocessed_tile_schema


# Define your existing functions here
def get_parent_tile(x: int, y: int, current_z: int = 16, parent_z: int = 14):
    assert current_z > parent_z, "Current zoom must be greater than parent zoom."
    zoom_diff = current_z - parent_z
    parent_x = x // (2**zoom_diff)
    parent_y = y // (2**zoom_diff)
    return parent_x, parent_y


def load_embeddings(args, cls_path, feat_path=None, limit=None):
    model = EmbeddingLookup(cls_path.replace("/cls.h5", ""), cls_only=True)
    embeddings_dict = {}

    id_path = cls_path.replace("cls.h5", "identifiers.h5")
    with h5py.File(id_path, "r") as f:
        ids = f["identifiers"][:]

    if isinstance(ids[0], bytes):
        ids = [id.decode("utf-8") for id in ids]
    else:
        ids = [str(id) for id in ids]

    # random sort with seed
    np.random.seed(42)
    np.random.shuffle(ids)

    if limit is not None:
        ids = ids[:limit]

    if args.only:
        selected_tiles = args.only.split(",")
        for tile in selected_tiles:
            ids.append(tile)

    id_map = {tile_name: idx for idx, tile_name in enumerate(ids)}
    print(f"Running with {len(id_map)} tiles")

    # if feat_path and os.path.exists(feat_path):
    #     with h5py.File(feat_path, 'r') as feat_h5:
    #         print(f"Loading features from {feat_path}")
    #         for group_name in feat_h5:
    #             embeddings_dict[group_name] = feat_h5[group_name]['features'][:]
    #             print(f"  features data shape: {embeddings_dict[group_name].shape}: {group_name}")
    #             print(f"  features data shape: {embeddings_dict[group_name][1].shape}: {group_name}")
    #     use_features = True
    # else:
    with h5py.File(cls_path, "r") as cls_h5:
        print(f"Loading cls from {cls_path}")
        for tile_name, id in id_map.items():
            embeddings_dict[tile_name] = model(FakeTileBatch(tile_name))[0]
            # embeddings_dict[tile_name] = cls_h5['cls'][id][:]
        print(f"  cls tokens available: {len(embeddings_dict)}")

    if embeddings_dict:
        first_key = next(iter(embeddings_dict))
        embedding_dim = embeddings_dict[first_key].shape[-1]
        print(f"Embedding dimension: {embedding_dim}")
    else:
        raise ValueError("No embeddings found in the provided HDF5 files.")

    return embeddings_dict, embedding_dim


def compute_embeddings(embeddings_dict, shuffle=True):
    # Extract group names and stack embeddings
    group_names = list(embeddings_dict.keys())
    all_embeddings = np.stack([embeddings_dict[name] for name in group_names])

    # Initialize and fit PaCMAP model
    # pacmap_model = PaCMAP(n_components=2)
    # reduced = pacmap_model.fit_transform(all_embeddings)
    reduced = np.arange(len(group_names))

    # Shuffle the data if required
    if shuffle:
        np.random.seed(42)
        permutation = np.random.permutation(len(group_names))
        group_names = [group_names[i] for i in permutation]
        all_embeddings = all_embeddings[permutation]
        # reduced = reduced[permutation]

    return group_names, all_embeddings, reduced


image_dict = {}


def load_image(tile_name, image_dir, return_np=False):
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
            if return_np:
                image_dict[f"16_{x}_{y}"] = np_img
            else:
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

    return image_dict[tile_name]


def init_embeddings(args):
    # Load embeddings

    cls_h5_path = args.embedding_dir + "/cls.h5"  # Update with actual path
    feat_h5_path = None  # Update with actual path or None
    image_dir = args.image_dir  # Update with actual path

    embeddings_dict, embedding_dim = load_embeddings(
        args, cls_h5_path, feat_h5_path, limit=args.limit
    )
    tile_names, all_embeddings, reduced = compute_embeddings(
        embeddings_dict, shuffle=args.shuffle
    )
    knn_model = NearestNeighbors(
        n_neighbors=args.n_neighbours + 1, algorithm="auto"
    ).fit(all_embeddings.tolist())

    return {
        "tile_names": tile_names,
        "tile_embedding_dict": embeddings_dict,
        "all_embeddings": all_embeddings.tolist(),
        "reduced": reduced.tolist(),
        "knn_model": knn_model,
    }


geometry_image_cache = {}
pbf_tile_cache = {}


def get_pbf_tile(name):
    if name not in pbf_tile_cache:
        tile_coords = list(map(int, name.split("_")))
        px, py = get_parent_tile(tile_coords[1], tile_coords[2])
        parent_name = f"14_{px}_{py}"
        tile_group = unprocessed_tile_schema.TileGroup()
        with open(f"data/tiles/huge/merged/{parent_name}.pbf", "rb") as f:
            tile_group.ParseFromString(f.read())
        for tile in tile_group.tiles:
            pbf_tile_cache[f"16_{tile.x}_{tile.y}"] = tile

    return pbf_tile_cache[name]


tag_indexer = None


def get_tile_tags(name, top=5):
    global tag_indexer
    if tag_indexer is None:
        tag_indexer = TagIndexer("data/tiles/embeddings.pkl")
    tile = get_pbf_tile(name)
    tag_count = defaultdict(int)
    for f in tile.features:
        for k, v in f.tags.items():
            k, v = tag_indexer.index_to_tag(tag_indexer.tag_to_index((k, v)))
            tag_count[k, v] += 1

    # get top5
    tags = dict(sorted(tag_count.items(), key=lambda x: x[1], reverse=True)[:5])
    display = [f"{count}x [{k}={v}]" for (k, v), count in tags.items()]
    not_covered = len(tag_count) - len(display)
    if not_covered > 0:
        display.append(f"and {not_covered} more..")
    return display


def create_feature_geometry_image(feature):
    fig, ax = plt.subplots()
    coords = feature.geometry.points

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


def create_tile_geometry_image(tile):
    fig, ax = plt.subplots()
    fids_in_groups = set()
    for g in tile.groups:
        for fid in g.feature_indices:
            fids_in_groups.add(fid)
    for fid, feature in enumerate(tile.features):
        geo = feature.geometry
        x = [c.lon for c in geo.points]
        y = [c.lat for c in geo.points]

        color = None
        if fid in fids_in_groups:
            color = "green"

        if geo.is_closed and len(geo.points) > 2:
            xy = np.array(list(zip(x, y)))
            if xy.ndim != 2:
                print("WARNING: Skipping invalid geometry")
                continue
            area = np.abs(np.sum(xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1])) / 2
            lerp = lambda a, b, t: (1 - t) * a + t * b
            clamp = lambda x: max(0, min(1, x))
            t = clamp((area - 1e-10) / (1e-6 - 1e-10))
            alpha = lerp(0.9, 0.2, t)
            polygon = plt.Polygon(
                xy,
                closed=True,
                fill=True,
                color=color if color else "blue",
                alpha=alpha,
                zorder=0 if color else 3,
            )
            ax.add_patch(polygon)
        else:
            if len(geo.points) == 1:
                ax.scatter(x, y, color=color if color else "black", s=10, zorder=10)
            else:
                ax.plot(x, y, color=color if color else "orange", linewidth=2, zorder=5)

    ax.set_aspect("equal")
    (lat1, lon1, lat2, lon2) = tile_zxy_to_latlon_bbox(tile.zoom, tile.x, tile.y)
    ax.set_xlim(lon1, lon2)
    ax.set_ylim(lat1, lat2)
    ax.axis("off")  # Hide axes

    # Save to a bytes buffer
    # plt.show()
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def get_geometry_image(name):
    if name in geometry_image_cache:
        return geometry_image_cache[name]

    tile = get_pbf_tile(name)
    img_str = create_tile_geometry_image(tile)
    geometry_image_cache[name] = img_str
    return img_str


def get_dist(name1, name2):
    coords1 = list(map(int, name1.split("_")))
    coords2 = list(map(int, name2.split("_")))
    return np.linalg.norm(np.array(coords1) - np.array(coords2))


def generate_tile_table_html(file_name, data, args):
    tile_names = data["tile_names"]
    tile_embedding_dict = data["tile_embedding_dict"]

    # Pick n random tile names
    n = args.num_tiles
    if n > len(tile_names):
        n = len(tile_names)
    random_tile_names = tile_names[:n]

    if args.only:
        random_tile_names = args.only.split(",")

    pretty = args.pretty

    display_header = (
        lambda x: f"""<td><div style="display:flex;"><strong style="margin:auto;">{x}</strong></div></td>"""
    )

    def display_image(img, name, border=False):
        img_style = "border-radius:5px;"
        img_h = 224
        img_w = 224
        geo_style = "border-style: solid; border-color: rgb(0,0,0,0.6); border-width: 1px; border-radius: 5px;"
        if border:
            img_style += "border-style: solid; border-color: red; border-width: 2px;"
            img_h = 220
            img_w = 220
        return f"""
            <td style="padding: 5px;">
                <img src="{img}" width="{img_w}" height="{img_h}" alt="Neighbour Image {j + 1}" style="{img_style}"/>
                <div style="height: 224px; margin-top: 6px;">
                    <img src="{get_geometry_image(name)}" width="222" height="222" alt="Tile Image" style="{geo_style}"/>
                </div>
            </td>
        """

    display_tags = (
        lambda name: f"""
            <td style="padding-left: 20px; overflow-x:hidden; word-break: break-all;">
                {"".join([f"<strong>{tag}</strong><br/>" for tag in get_tile_tags(name)])}
            </td>
    """
    )
    # Build a simple HTML table
    table_rows = []
    knn_model = data["knn_model"]
    for tile_name in tqdm(
        random_tile_names, desc="Rendering", total=len(random_tile_names)
    ):
        tile_img = load_image(tile_name, args.image_dir)
        embedding = tile_embedding_dict[tile_name].reshape(1, -1)
        distances, indices = knn_model.kneighbors(embedding)
        distances = distances[0]  # Exclude self
        indices = indices[0]
        neighbors = [tile_names[idx] for idx in indices]
        self_idx = neighbors.index(tile_name)
        neighbors.pop(self_idx)
        distances = np.delete(distances, self_idx)

        neighbours_html = []
        for j, neighbour in enumerate(neighbors):
            neighbour_img = load_image(neighbour, args.image_dir)
            neighbours_html.append(display_image(neighbour_img, neighbour))
        row_html = f"""
        <tr>
            {display_header("Query")}
            {"".join([display_header(f"N {j + 1}") for j, n in enumerate(neighbors)])}
        </tr>
        <tr>
            {display_image(tile_img, tile_name, border=True)}
            {"".join(neighbours_html)}
        </tr>
        <tr>
            {display_tags(tile_name)}
            {"".join([display_tags(n) for j, n in enumerate(neighbors)])}
        </tr>
        <tr>
            <td style="padding: 5px;">
                <p>{tile_name}<p>
            </td>
            {
            "".join(
                [
                    f"<td style='padding: 5px;'>"
                    f"<p>tile: {n}</p>"
                    f"<p>emb_dist: {distances[j]:.1f}</p>"
                    f"<p>tile_dist: {get_dist(tile_name, n):.0f}</p>"
                    "</td>"
                    for j, n in enumerate(neighbors)
                ]
            )
        }
        </tr>
        """
        table_rows.append(row_html)

    table_html = f"""
    <table cellspacing="0" cellpadding="0" style="border-collapse:collapse;border:none;">
      <tr>
        <th style="padding: 5px;">Tile Name</th>
        <th style="padding: 5px;">Image</th>
      </tr>
      {"".join(table_rows)}
    </table>
    """

    # Wrap in a basic HTML page
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{file_name}</title>
    </head>
    <body>
        <h1>Displaying {n} Random Tiles</h1>
        <h2>{file_name}</h2>
        {table_html}
    </body>
    </html>
    """

    return full_html


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform kNN search on HDF5 embeddings with image visualization, "
        "and generate an HTML file for a random subset."
    )
    parser.add_argument(
        "-e",
        "--embedding-dir",
        type=str,
        required=True,
        help="Path to the directory containing cls.h5 and identifiers.h5",
    )
    parser.add_argument(
        "-img",
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing group images (.webp files)",
    )
    # New argument:
    parser.add_argument(
        "-n",
        "--num-tiles",
        type=int,
        default=50,
        help="Number of random tiles to display in the HTML output",
    )
    parser.add_argument(
        "-k",
        "--n_neighbours",
        type=int,
        default=4,
        help="Number of nearest neighbours to display",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Number of embeddings to read"
    )
    parser.add_argument("--name", type=str, default=None, help="Name of file")
    parser.add_argument(
        "--shuffle", type=bool, default=True, help="Disable shuffling of embeddings"
    )
    parser.add_argument("--pretty", type=bool, default=True)
    parser.add_argument("--only", type=str, default=None)

    args = parser.parse_args()
    embedding_dir_name = (
        args.name if args.name else "-".join(args.embedding_dir.split("/")[-2:])
    )

    # Step 1: Load all embeddings, images, etc.
    data = init_embeddings(args)

    # Step 2: Generate an HTML table for n random tiles
    html_output = generate_tile_table_html(embedding_dir_name, data, args)

    # Step 3: Write the HTML to a file
    output_dir = "pyscripts/out/knn/"

    makedirs(output_dir, exist_ok=True)

    output_filename = f"{output_dir}/{embedding_dir_name}.html"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_output)

    # if is mac
    if os.name == "posix":
        os.system(f"open '{output_filename}'")

    print(f"HTML file successfully generated: {output_filename}")
