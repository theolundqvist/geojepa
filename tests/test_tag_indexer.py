import pandas as pd
import numpy as np
import torch


def create_mock_embeddings(file_path: str):
    data = {
        "key": ["color", "color", "shape"],
        "value": ["red", "blue", "circle"],
        "embedding": [
            torch.ones(1024),
            torch.ones(1024) * 2,
            torch.ones(1024) * 3,
        ],
    }
    df = pd.DataFrame(data)
    df.to_pickle(file_path)


import pytest

from src.modules.tag_encoder import (
    TagIndexer,
    TagEncoder,
)  # Replace with your actual module name


@pytest.fixture(scope="module")
def mock_embedding_file(tmp_path_factory):
    file_path = tmp_path_factory.mktemp("data") / "mock_embeddings.pkl"
    create_mock_embeddings(file_path)
    return file_path


def test_tag_to_index(mock_embedding_file):
    indexer = TagIndexer(embedding_file=str(mock_embedding_file))

    # Define expected tags and their indices
    expected_tags = {
        ("color", "red"): 0,
        ("color", "blue"): 1,
        ("shape", "circle"): 2,
    }

    for tag, expected_index in expected_tags.items():
        assert indexer.tag_to_index(tag) == expected_index, (
            f"Tag {tag} should map to index {expected_index}"
        )

    # Test a tag that does not exist
    assert indexer.tag_to_index(("nonexistent", "tag")) == -1


def test_index_to_tag(mock_embedding_file):
    indexer = TagIndexer(embedding_file=str(mock_embedding_file))

    # Define expected indices and their tags
    expected_indices = {
        0: ("color", "red"),
        1: ("color", "blue"),
        2: ("shape", "circle"),
    }

    for index, expected_tag in expected_indices.items():
        assert indexer.index_to_tag(index) == expected_tag, (
            f"Index {index} should map to tag {expected_tag}"
        )

    # Test an index that does not exist
    assert indexer.index_to_tag(999) == ("", "")


def test_index_to_embedding(mock_embedding_file):
    encoder = TagEncoder(embedding_file=str(mock_embedding_file))

    # Manually load the embeddings to compare
    with open(mock_embedding_file, "rb") as f:
        loaded_embeddings = (
            pd.read_pickle(f).set_index(["key", "value"])["embedding"].to_dict()
        )

    for i, (tag, embedding) in enumerate(loaded_embeddings.items()):
        assert torch.allclose(encoder.index_to_embedding[i], embedding), (
            f"Embedding at index {i} does not match"
        )


def test_consistent_loading(mock_embedding_file):
    # Load TagIndexer and TagEncoder multiple times and verify consistency
    indexer1 = TagIndexer(embedding_file=str(mock_embedding_file))
    indexer2 = TagIndexer(embedding_file=str(mock_embedding_file))

    assert indexer1.tag_to_index_dict == indexer2.tag_to_index_dict, (
        "TagIndexer indices should be consistent across loads"
    )
    assert indexer1.index_to_tag_dict == indexer2.index_to_tag_dict, (
        "TagIndexer tag mappings should be consistent across loads"
    )

    encoder1 = TagEncoder(embedding_file=str(mock_embedding_file))
    encoder2 = TagEncoder(embedding_file=str(mock_embedding_file))

    for i in encoder1.index_to_embedding:
        assert torch.allclose(
            encoder1.index_to_embedding[i], encoder2.index_to_embedding[i]
        ), f"Embeddings at index {i} should be consistent across loads"


def test_missing_values_handling(mock_embedding_file, tmp_path):
    # Create a pickle file with NaN values
    data = {
        "key": ["color", "color", "shape"],
        "value": ["red", np.nan, "circle"],
        "embedding": [
            torch.ones(1024),
            torch.ones(1024) * 2,
            torch.ones(1024) * 3,
        ],
    }
    df = pd.DataFrame(data)
    nan_file = tmp_path / "nan_embeddings.pkl"
    df.to_pickle(nan_file)

    indexer = TagIndexer(embedding_file=str(nan_file))

    # The NaN value should be replaced with an empty string
    assert indexer.tag_to_index(("color", "")) == 1, (
        "NaN value should be replaced with an empty string and correctly indexed"
    )

    # Verify that the index to tag mapping works correctly
    assert indexer.index_to_tag(1) == ("color", ""), (
        "Index 1 should map to ('color', '')"
    )
