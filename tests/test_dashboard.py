import os
import pytest
import numpy as np
import torch
from torch_geometric.data import Data

from src.dashboard.dashboard_utils import (
    load_preprocessing_metadata,
    load_graph_split,
    load_training_history,
    build_interactive_graph_figure,
    plot_training_history_curves,
    create_attention_radar_figure,
    CATEGORY_COLORS
)


def test_load_preprocessing_metadata():
    """Verify loading of schema and mappings."""
    meta = load_preprocessing_metadata()
    assert "num_features" in meta
    assert meta["num_features"] == 45
    assert len(meta["features"]) == 45


def test_load_graph_split():
    """Verify loading of existing graph artifacts."""
    train_data, train_meta = load_graph_split(split="train")
    if train_data is not None:
        assert isinstance(train_data, Data)
        assert train_data.num_nodes > 0
        assert train_data.x.shape[1] == 45
        assert train_meta is not None
        assert "num_edges" in train_meta

    # Test non-existent split
    missing_data, missing_meta = load_graph_split(split="nonexistent_split")
    assert missing_data is None
    assert missing_meta is None


def test_load_training_history():
    """Verify loading of training history JSON."""
    hist = load_training_history()
    if hist is not None:
        assert "config" in hist
        assert "history" in hist
        assert "best_epoch" in hist


def test_build_interactive_graph_figure():
    """Verify interactive Plotly network graph generation."""
    x = torch.randn(10, 45, dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    y = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=10)
    data.label_category = ["Benign", "ARP Spoofing", "MQTT attacks", "Reconnaissance", "TCP/IP DoS/DDoS"] * 2
    data.label_subtype = ["Benign"] * 10

    fig = build_interactive_graph_figure(data, max_nodes=10, highlight_node=0, neighbor_indices=[1])
    assert fig is not None
    assert len(fig.data) > 0


def test_plot_training_history_curves():
    """Verify Plotly convergence curves."""
    sample_hist = {
        "best_epoch": 2,
        "history": {
            "train_loss": [1.0, 0.5, 0.3],
            "val_loss": [1.1, 0.6, 0.4],
            "val_accuracy": [0.6, 0.8, 0.9],
            "val_macro_f1": [0.55, 0.78, 0.88]
        }
    }
    fig = plot_training_history_curves(sample_hist)
    assert fig is not None
    assert len(fig.data) == 4


def test_create_attention_radar_figure():
    """Verify multi-head attention radar plot generation."""
    top_neighbors = [
        {
            "rank": 1,
            "neighbor_index": 5,
            "neighbor_category": "MQTT attacks",
            "layer_1_heads": [0.4, 0.2, 0.3, 0.1],
            "combined_attention": 0.45
        }
    ]
    fig = create_attention_radar_figure(top_neighbors)
    assert fig is not None
    assert len(fig.data) == 1
