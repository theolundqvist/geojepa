import pytest

from src.utils.tile_utils import get_sub_tiles, get_parent_tile


@pytest.mark.parametrize("x", [10, 1000, 10000])
@pytest.mark.parametrize("y", [10, 1000, 10000])
def test_tile_zoom_conversions(x, y):
    """Test that the loader returns the expected batch size."""
    tiles = get_sub_tiles(x, y, 14, 16)
    assert len(tiles) == 16
    for sub_x, sub_y in tiles:
        assert get_parent_tile(sub_x, sub_y, 16, 14) == (x, y)


def test_edge_case():
    tiles = get_sub_tiles(2620, 6339, 14, 16)
    print(tiles)
    assert len(tiles) == 16
    for x, y in tiles:
        assert get_parent_tile(x, y, 16, 14) == (2620, 6339)

    assert (10482, 25360) not in tiles

    assert get_parent_tile(10482, 25360, 16, 14) != (2620, 6339)
