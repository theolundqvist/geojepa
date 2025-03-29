from typing import Tuple, List


def get_parent_tile(x, y, current_z=16, parent_z=14) -> Tuple[int, int]:
    assert current_z > parent_z
    # zoom 16 -> zoom 14
    zoom_diff = current_z - parent_z
    # Convert coordinates
    parent_x = x // (2**zoom_diff)
    parent_y = y // (2**zoom_diff)
    return parent_x, parent_y


def get_sub_tiles(x, y, current_z=14, sub_z=16) -> List[Tuple[int, int]]:
    assert sub_z > current_z
    new_coords = []
    zoom_diff = sub_z - current_z
    sub_x = x * (2**zoom_diff)
    sub_y = y * (2**zoom_diff)
    for j in range(4):
        for i in range(4):
            new_coords.append((sub_x + i, sub_y + j))
    return new_coords


import math


def tile_zxy_to_latlon_bbox(z: int, x: int, y: int) -> tuple:
    """
    Converts tile coordinates (z, x, y) to latitude and longitude bounding box.

    Args:
        z (int): Zoom level.
        x (int): Tile x coordinate.
        y (int): Tile y coordinate.

    Returns:
        tuple: (lat1, lon1, lat2, lon2) representing the bounding box.
    """
    MIN_ZOOM_LEVEL = 0
    MAX_ZOOM_LEVEL = 22

    if not (MIN_ZOOM_LEVEL <= z <= MAX_ZOOM_LEVEL):
        raise ValueError(
            f"Zoom level {z} is out of range [{MIN_ZOOM_LEVEL}, {MAX_ZOOM_LEVEL}]"
        )

    max_xy = 2**z - 1
    if not (0 <= x <= max_xy):
        raise ValueError(f"Tile x value {x} is out of range [0, {max_xy}]")

    if not (0 <= y <= max_xy):
        raise ValueError(f"Tile y value {y} is out of range [0, {max_xy}]")

    def tile_to_lon(x, z):
        return x / (2**z) * 360.0 - 180.0

    def tile_to_lat(y, z):
        n = math.pi - 2.0 * math.pi * y / (2**z)
        return math.degrees(math.atan(0.5 * (math.exp(n) - math.exp(-n))))

    lon1 = tile_to_lon(x, z)
    lat1 = tile_to_lat(y, z)
    lon2 = tile_to_lon(x + 1, z)
    lat2 = tile_to_lat(y + 1, z)

    return (min(lat1, lat2), min(lon1, lon2), max(lat1, lat2), max(lon1, lon2))
