from src.modules.tag_encoder import TagEncoder, TagIndexer
from src.modules.tag_models import Tagformer


def test_tag_transformer():
    encoder = Tagformer(embedding_file="data/tiles/embeddings.pkl", out_dim=128)


def test_tag_encoder() -> None:
    encoder = TagEncoder()
    x = encoder.forward(torch.tensor([[[0]]]))
    assert x.abs().sum() < 1e-9
    x = encoder.forward(torch.tensor([[[2, 4, 5], [2, 1232, 4000]]]))
    print(x.mean(), x.std(), x.abs().sum())


import pytest
import pandas as pd
import torch
import numpy as np


@pytest.fixture
def create_temp_pickle(tmp_path):
    """
    Fixture to create a temporary pickle file with specified data.
    Returns the path to the pickle file.
    """

    def _create_temp_pickle(data: pd.DataFrame):
        file_path = tmp_path / "embeddings.pkl"
        data["embedding"] = data["embedding"].apply(lambda x: x.to(torch.float))
        data.to_pickle(file_path)
        return file_path

    return _create_temp_pickle


def test_initialization_success(create_temp_pickle):
    """Test that TagIndexer initializes correctly with valid embeddings."""
    # Prepare mock embeddings data
    data = pd.DataFrame(
        {
            "key": ["key1", "key2", "key3"],
            "value": ["val1", "val2", "nan"],
            "embedding": [torch.ones(5), torch.ones(5) * 2, torch.ones(5) * 3],
        }
    )

    # Create temporary pickle file
    pickle_path = create_temp_pickle(data)

    # Initialize TagIndexer
    tag_indexer = TagIndexer(embedding_file=str(pickle_path), dim=5)

    # Check loaded_embeddings
    assert (
        tag_indexer.loaded_embeddings[("key1", "val1")].tolist()
        == torch.ones(5).tolist()
    )
    assert (
        tag_indexer.loaded_embeddings[("key2", "val2")].tolist()
        == (torch.ones(5) * 2).tolist()
    )
    assert (
        tag_indexer.loaded_embeddings[("key3", "")].tolist()
        == (torch.ones(5) * 3).tolist()
    )
    assert np.array_equal(tag_indexer.loaded_embeddings[("PAD", "PAD")], np.zeros(5))

    # Check tag_to_index_dict
    expected_tag_to_index = {
        ("PAD", "PAD"): 0,
        ("PAD", ""): 0,
        ("key1", "val1"): 1,
        ("key2", "val2"): 2,
        ("key3", ""): 3,  # 'nan' converted to ''
        ("key1", ""): 1,
        ("key2", ""): 2,
        ("key3", ""): 3,  # 'nan' converted to ''
    }
    assert tag_indexer.tag_to_index_dict == expected_tag_to_index

    # Check index_to_tag_dict
    expected_index_to_tag = {
        1: ("key1", "val1"),
        2: ("key2", "val2"),
        3: ("key3", ""),
        0: ("PAD", "PAD"),
    }
    assert tag_indexer.index_to_tag_dict == expected_index_to_tag


def test_initialization_file_not_found(tmp_path):
    """Test that TagIndexer raises an assertion error if embedding file does not exist."""
    non_existent_path = tmp_path / "non_existent.pkl"
    with pytest.raises(
        AssertionError, match=f"Embedding file not found at {non_existent_path}"
    ):
        TagIndexer(embedding_file=str(non_existent_path), dim=5)


def test_tag_to_index(create_temp_pickle):
    """Test the tag_to_index method."""
    # Prepare mock embeddings data
    data = pd.DataFrame(
        {
            "key": ["key1", "key2", "key3"],
            "value": ["val1", "val2", "nan"],
            "embedding": [torch.ones(5), torch.ones(5) * 2, torch.ones(5) * 3],
        }
    )

    # Create temporary pickle file
    pickle_path = create_temp_pickle(data)

    # Initialize TagIndexer
    tag_indexer = TagIndexer(embedding_file=str(pickle_path), dim=5)

    # Test existing tags
    assert tag_indexer.tag_to_index(("key1", "val1")) == 1
    assert tag_indexer.tag_to_index(("key2", "val2")) == 2
    assert (
        tag_indexer.tag_to_index(("key3", "val3")) == 3
    )  # 'val3' treated as 'nan' -> 3

    # Test unknown tag defaults to PAD index
    assert tag_indexer.tag_to_index(("unknown", "unknown")) == 0


def test_index_to_tag(create_temp_pickle):
    """Test the index_to_tag method."""
    # Prepare mock embeddings data
    data = pd.DataFrame(
        {
            "key": ["key1", "key2", "key3"],
            "value": ["val1", "val2", "nan"],
            "embedding": [torch.ones(5), torch.ones(5) * 2, torch.ones(5) * 3],
        }
    )

    # Create temporary pickle file
    pickle_path = create_temp_pickle(data)

    # Initialize TagIndexer
    tag_indexer = TagIndexer(embedding_file=str(pickle_path), dim=5)

    # Test valid indices
    assert tag_indexer.index_to_tag(1) == ("key1", "val1")
    assert tag_indexer.index_to_tag(2) == ("key2", "val2")
    assert tag_indexer.index_to_tag(3) == ("key3", "")
    assert tag_indexer.index_to_tag(0) == ("PAD", "PAD")

    # Test invalid index
    assert tag_indexer.index_to_tag(999) == ("", "")  # Non-existent index


def test_index_to_embedding(create_temp_pickle):
    """Test the index_to_embedding method."""
    # Prepare mock embeddings data
    data = pd.DataFrame(
        {
            "key": ["key1", "key2", "key3"],
            "value": ["val1", "val2", "nan"],
            "embedding": [torch.ones(5), torch.ones(5) * 2, torch.ones(5) * 3],
        }
    )

    # Create temporary pickle file
    pickle_path = create_temp_pickle(data)

    # Initialize TagIndexer
    tag_indexer = TagIndexer(embedding_file=str(pickle_path), dim=5)

    # Test existing embeddings
    assert torch.equal(tag_indexer.index_to_embedding(1), torch.ones(5))
    assert torch.equal(tag_indexer.index_to_embedding(2), torch.ones(5) * 2)
    assert torch.equal(tag_indexer.index_to_embedding(3), torch.ones(5) * 3)

    # Test PAD embedding
    pad_embedding = tag_indexer.index_to_embedding(0)
    assert isinstance(pad_embedding, torch.Tensor)
    assert torch.equal(pad_embedding, torch.tensor(np.zeros(5)))

    # Test default embedding for unknown index
    default_embedding = tag_indexer.index_to_embedding(999)
    assert torch.equal(default_embedding, torch.tensor(np.zeros(5)))


def test_embedding_matrix(create_temp_pickle):
    """Test the embedding_matrix method."""
    # Prepare mock embeddings data
    data = pd.DataFrame(
        {
            "key": ["key1", "key2", "key3"],
            "value": ["val1", "val2", "nan"],
            "embedding": [torch.ones(5), torch.ones(5) * 2, torch.ones(5) * 3],
        }
    )

    # Create temporary pickle file
    pickle_path = create_temp_pickle(data)

    # Initialize TagIndexer
    tag_indexer = TagIndexer(embedding_file=str(pickle_path), dim=5)

    # Get embedding matrix
    embedding_matrix = tag_indexer.embedding_matrix()

    # Expected embedding matrix shape: (4, 5)
    assert embedding_matrix.shape == (4, 5)

    # Check specific embeddings
    assert torch.equal(embedding_matrix[1], torch.ones(5))
    assert torch.equal(embedding_matrix[2], torch.ones(5) * 2)
    assert torch.equal(embedding_matrix[3], torch.ones(5) * 3)
    assert torch.equal(embedding_matrix[0], torch.tensor(np.zeros(5)))


def test_handling_nan_values(create_temp_pickle):
    """Test that 'nan' values in 'value' are handled correctly."""
    # Prepare mock embeddings data with 'nan'
    data = pd.DataFrame(
        {
            "key": ["key1", "key2", "key3"],
            "value": ["val1", "val2", "nan"],
            "embedding": [torch.ones(5), torch.ones(5) * 2, torch.ones(5) * 3],
        }
    )

    # Create temporary pickle file
    pickle_path = create_temp_pickle(data)

    # Initialize TagIndexer
    tag_indexer = TagIndexer(embedding_file=str(pickle_path), dim=5)

    # Check that 'nan' is converted to ''
    assert tag_indexer.tag_to_index(("key3", "ASDNALSDH")) == 3
    assert tag_indexer.tag_to_index(("key3", "")) == 3


def test_padding_embedding(create_temp_pickle):
    """Test that the padding embedding is correctly added."""
    # Prepare mock embeddings data
    data = pd.DataFrame(
        {"key": ["key1"], "value": ["val1"], "embedding": [torch.ones(5)]}
    )

    # Create temporary pickle file
    pickle_path = create_temp_pickle(data)

    # Initialize TagIndexer
    tag_indexer = TagIndexer(embedding_file=str(pickle_path), dim=5)

    # Retrieve PAD embedding
    pad_embedding = tag_indexer.index_to_embedding(0)
    assert isinstance(pad_embedding, torch.Tensor)
    assert torch.equal(pad_embedding, torch.tensor(np.zeros(5)))


def test_embedding_data_types(create_temp_pickle):
    """Test that embeddings are of correct data types after initialization."""
    # Prepare mock embeddings data
    data = pd.DataFrame(
        {
            "key": ["key1", "key2"],
            "value": ["val1", "val2"],
            "embedding": [torch.ones(5), torch.ones(5) * 2],
        }
    )

    # Create temporary pickle file
    pickle_path = create_temp_pickle(data)

    # Initialize TagIndexer
    tag_indexer = TagIndexer(embedding_file=str(pickle_path), dim=5)

    # Check data types
    for emb in tag_indexer.loaded_embeddings.values():
        assert emb.dtype in [torch.float]
