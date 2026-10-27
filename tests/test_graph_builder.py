import os
import pytest
import numpy as np
import torch
from torch_geometric.data import Data

from src.data.preprocess import EXPECTED_FEATURES
from src.graph.graph_builder import (
    GraphBuildConfig,
    build_flow_similarity_graph,
    save_graph_artifacts,
    load_graph_artifact
)


@pytest.fixture
def synthetic_features():
    """Generates synthetic standardized features for 50 nodes and 45 features."""
    np.random.seed(42)
    return np.random.randn(50, 45).astype(np.float32)


@pytest.fixture
def synthetic_labels():
    """Generates synthetic labels matching 50 nodes."""
    return {
        "category_id": np.random.randint(0, 5, size=50, dtype=np.int64),
        "subtype_id": np.random.randint(0, 19, size=50, dtype=np.int64),
        "is_attack": np.random.randint(0, 2, size=50, dtype=np.int64),
        "label_raw": [f"flow_{i}" for i in range(50)],
        "label_subtype": ["MQTT-DDoS-Connect_Flood" if i % 2 == 0 else "Benign" for i in range(50)],
        "label_category": ["MQTT attacks" if i % 2 == 0 else "Benign" for i in range(50)],
    }


def test_graph_dimensions_and_features(synthetic_features, synthetic_labels):
    """Verify that graph tensor shapes match exact node count and 45 features."""
    config = GraphBuildConfig(k_neighbors=5)
    data = build_flow_similarity_graph(synthetic_features, labels=synthetic_labels, config=config)

    assert data.num_nodes == 50
    assert data.x.shape == (50, 45)
    assert data.x.dtype == torch.float32
    assert data.edge_index.dtype == torch.long
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] > 0


def test_valid_edge_indices(synthetic_features):
    """Verify that every edge connects valid node indices [0, N-1]."""
    config = GraphBuildConfig(k_neighbors=5)
    data = build_flow_similarity_graph(synthetic_features, config=config)

    src, dst = data.edge_index[0], data.edge_index[1]
    assert (src >= 0).all() and (src < data.num_nodes).all()
    assert (dst >= 0).all() and (dst < data.num_nodes).all()


def test_no_self_neighbors_in_knn(synthetic_features):
    """Verify that no node connects to itself in the constructed k-NN graph."""
    for k in (3, 5, 8):
        config = GraphBuildConfig(k_neighbors=k)
        data = build_flow_similarity_graph(synthetic_features, config=config)
        self_loops = (data.edge_index[0] == data.edge_index[1])
        assert not self_loops.any(), f"Found self-loop in graph with k={k}"


def test_deterministic_graph_generation(synthetic_features, synthetic_labels):
    """Verify that repeated graph construction with the same inputs produces identical graphs."""
    config = GraphBuildConfig(k_neighbors=5, random_seed=42)
    data1 = build_flow_similarity_graph(synthetic_features, labels=synthetic_labels, config=config)
    data2 = build_flow_similarity_graph(synthetic_features, labels=synthetic_labels, config=config)

    assert torch.equal(data1.x, data2.x)
    assert torch.equal(data1.edge_index, data2.edge_index)
    assert torch.equal(data1.y, data2.y)


def test_correct_label_alignment(synthetic_features, synthetic_labels):
    """Verify that node labels and metadata align accurately with input data."""
    config = GraphBuildConfig(target_label="category_id")
    data = build_flow_similarity_graph(synthetic_features, labels=synthetic_labels, config=config)

    expected_y = torch.tensor(synthetic_labels["category_id"], dtype=torch.long)
    assert torch.equal(data.y, expected_y)
    assert torch.equal(data.category_id, expected_y)
    assert torch.equal(data.is_attack, torch.tensor(synthetic_labels["is_attack"], dtype=torch.long))
    assert data.label_category == synthetic_labels["label_category"]


def test_labels_and_metadata_excluded_from_x(synthetic_features, synthetic_labels):
    """Verify that x contains exclusively the 45 numerical features and no label columns."""
    config = GraphBuildConfig()
    data = build_flow_similarity_graph(synthetic_features, labels=synthetic_labels, config=config)

    # x must be strictly 45 features
    assert data.x.shape[1] == 45
    # Verify values in x match the input feature array exactly
    assert np.allclose(data.x.numpy(), synthetic_features)


def test_no_duplicate_edges(synthetic_features):
    """Verify that edge_index contains no duplicate edges in undirected and directed modes."""
    # Undirected mode
    config_undir = GraphBuildConfig(k_neighbors=5, make_undirected=True)
    data_undir = build_flow_similarity_graph(synthetic_features, config=config_undir)
    edges_undir = list(zip(data_undir.edge_index[0].tolist(), data_undir.edge_index[1].tolist()))
    assert len(edges_undir) == len(set(edges_undir)), "Found duplicate edges in undirected edge_index"

    # Directed mode
    config_dir = GraphBuildConfig(k_neighbors=5, make_undirected=False)
    data_dir = build_flow_similarity_graph(synthetic_features, config=config_dir)
    edges_dir = list(zip(data_dir.edge_index[0].tolist(), data_dir.edge_index[1].tolist()))
    assert len(edges_dir) == len(set(edges_dir)), "Found duplicate edges in directed edge_index"


def test_small_graphs_and_edge_cases():
    """Verify robust handling of empty graphs, single nodes, small k, and invalid inputs."""
    config = GraphBuildConfig(k_neighbors=5)

    # 1. Empty graph
    empty_feats = np.empty((0, 45), dtype=np.float32)
    data_empty = build_flow_similarity_graph(empty_feats, config=config)
    assert data_empty.num_nodes == 0
    assert data_empty.edge_index.shape == (2, 0)

    # 2. Single isolated node (must have no edges, correct label and metadata alignment)
    single_feat = np.ones((1, 45), dtype=np.float32)
    single_labels = {
        "category_id": [3],
        "subtype_id": [8],
        "is_attack": [1],
        "label_raw": ["recon_test"],
        "label_subtype": ["Recon-Port_Scan"],
        "label_category": ["Reconnaissance"],
        "source_row_index": [42],
    }
    data_single = build_flow_similarity_graph(single_feat, labels=single_labels, config=config)
    assert data_single.num_nodes == 1
    assert data_single.edge_index.shape == (2, 0)
    assert data_single.y.item() == 3
    assert data_single.label_category == ["Reconnaissance"]
    assert data_single.source_row_index.item() == 42

    # 3. Graph smaller than k (N=3, k=5 -> clamped to eff_k=2)
    small_feats = np.random.randn(3, 45).astype(np.float32)
    data_small = build_flow_similarity_graph(small_feats, config=config)
    assert data_small.num_nodes == 3
    # Each node connects to 2 other nodes -> undirected = 3 * 2 = 6 edges
    assert data_small.edge_index.shape[1] == 6

    # 4. Invalid k <= 0
    with pytest.raises(ValueError, match="k_neighbors must be a positive integer"):
        build_flow_similarity_graph(small_feats, config=GraphBuildConfig(k_neighbors=0))

    # 5. Invalid feature count (e.g. 40 instead of 45)
    invalid_feats = np.ones((10, 40), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected 45 features, got 40"):
        build_flow_similarity_graph(invalid_feats, config=config)

    # 6. NaN or Inf in features
    nan_feats = np.ones((5, 45), dtype=np.float32)
    nan_feats[0, 0] = np.nan
    with pytest.raises(ValueError, match="features matrix contains NaN or Inf"):
        build_flow_similarity_graph(nan_feats, config=config)


def test_inductive_isolation_no_test_to_train_leakage():
    """Verify that train and test graphs maintain strict topological isolation."""
    np.random.seed(101)
    train_feats = np.random.randn(40, 45).astype(np.float32)
    test_feats = np.random.randn(25, 45).astype(np.float32)

    config = GraphBuildConfig(k_neighbors=4)
    train_graph = build_flow_similarity_graph(train_feats, config=config)
    test_graph = build_flow_similarity_graph(test_feats, config=config)

    # Check node bounds
    assert train_graph.edge_index.max() < 40
    assert test_graph.edge_index.max() < 25

    # Completely separate graphs (independent node indices)
    assert train_graph.num_nodes == 40
    assert test_graph.num_nodes == 25


def test_save_and_load_graph_artifact(tmp_path, synthetic_features, synthetic_labels):
    """Verify artifact serialization (.pt and .json) and deserialization."""
    out_dir = str(tmp_path / "graphs")
    config = GraphBuildConfig(k_neighbors=5)
    data = build_flow_similarity_graph(synthetic_features, labels=synthetic_labels, config=config)
    meta = {"test_metric": 123, "nodes": data.num_nodes}

    pt_path, json_path = save_graph_artifacts(data, meta, out_dir, prefix="test_artifact")
    assert os.path.exists(pt_path)
    assert os.path.exists(json_path)

    loaded_data, loaded_meta = load_graph_artifact(pt_path, json_path)
    assert loaded_data.num_nodes == data.num_nodes
    assert torch.equal(loaded_data.x, data.x)
    assert torch.equal(loaded_data.edge_index, data.edge_index)
    assert torch.equal(loaded_data.y, data.y)
    assert loaded_meta["test_metric"] == 123
