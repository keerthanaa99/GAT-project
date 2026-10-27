import os
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.models.gat_model import GATClassifier, GATConfig
from src.models.train_gat import (
    create_stratified_node_split,
    compute_class_weights,
    evaluate_metrics,
    train_gat
)


@pytest.fixture
def sample_graph():
    """Creates a synthetic 20-node graph with 45 features and 5 classes."""
    torch.manual_seed(42)
    x = torch.randn(20, 45, dtype=torch.float32)
    # Simple cycle + random edges
    src = list(range(20)) + [0, 5, 10, 15]
    dst = list(range(1, 20)) + [0, 10, 15, 0, 5]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor([i % 5 for i in range(20)], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=20)


def test_gat_model_output_shapes(sample_graph):
    """Verify that GAT forward pass produces exact [N, num_classes] logit tensors."""
    config = GATConfig(in_dim=45, hidden_dim=16, num_classes=5, heads=2, num_layers=2)
    model = GATClassifier(config)
    model.eval()

    logits = model(sample_graph.x, sample_graph.edge_index)
    assert logits.shape == (20, 5)
    assert logits.dtype == torch.float32
    assert not torch.isnan(logits).any()


def test_gat_attention_weights_extraction(sample_graph):
    """Verify that multi-head attention weights and edge indices are correctly returned."""
    config = GATConfig(in_dim=45, hidden_dim=16, num_classes=5, heads=3, num_layers=2)
    model = GATClassifier(config)
    model.eval()

    logits, attentions = model(sample_graph.x, sample_graph.edge_index, return_attention_weights=True)

    assert logits.shape == (20, 5)
    assert len(attentions) == 2  # 2 layers

    # Layer 1 attention
    edge_idx_1, alpha_1 = attentions[0]
    assert edge_idx_1.shape[0] == 2
    assert alpha_1.shape[1] == 3  # 3 heads
    assert (alpha_1 >= 0.0).all() and (alpha_1 <= 1.0).all()

    # Final Layer attention (1 head)
    edge_idx_2, alpha_2 = attentions[1]
    assert edge_idx_2.shape[0] == 2
    assert alpha_2.shape[1] == 1  # 1 head
    assert (alpha_2 >= 0.0).all() and (alpha_2 <= 1.0).all()


def test_gat_predict_interface(sample_graph):
    """Verify the inference predict() method for probabilities and class predictions."""
    config = GATConfig(in_dim=45, hidden_dim=16, num_classes=5, heads=2)
    model = GATClassifier(config)

    res = model.predict(sample_graph.x, sample_graph.edge_index, return_attention_weights=True)

    probs = res["probabilities"]
    preds = res["predicted_classes"]

    assert probs.shape == (20, 5)
    assert preds.shape == (20,)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(20), atol=1e-5)
    assert ((preds >= 0) & (preds < 5)).all()
    assert "attentions" in res


def test_gat_checkpoint_save_and_load(tmp_path, sample_graph):
    """Verify model serialization and deserialization without numerical drift."""
    config = GATConfig(in_dim=45, hidden_dim=16, num_classes=5, heads=2)
    model = GATClassifier(config)
    model.eval()

    orig_logits = model(sample_graph.x, sample_graph.edge_index)

    checkpoint_file = str(tmp_path / "gat_checkpoint.pt")
    model.save_checkpoint(checkpoint_file, extra_metadata={"epoch": 10})
    assert os.path.exists(checkpoint_file)

    loaded_model = GATClassifier.from_checkpoint(checkpoint_file)
    loaded_logits = loaded_model(sample_graph.x, sample_graph.edge_index)

    assert torch.allclose(orig_logits, loaded_logits)


def test_gat_invalid_inputs(sample_graph):
    """Verify that invalid tensor shapes, NaNs, and bad configs raise informative errors."""
    config = GATConfig(in_dim=45, num_classes=5)
    model = GATClassifier(config)

    # 1. Feature dimension mismatch (40 instead of 45)
    with pytest.raises(ValueError, match="Expected x of shape"):
        model(torch.randn(20, 40), sample_graph.edge_index)

    # 2. Edge index dimension mismatch
    with pytest.raises(ValueError, match="Expected edge_index of shape"):
        model(sample_graph.x, torch.tensor([0, 1, 2]))

    # 3. NaNs in input features
    nan_x = sample_graph.x.clone()
    nan_x[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Input feature tensor contains NaN or Inf"):
        model(nan_x, sample_graph.edge_index)

    # 4. Invalid config (num_layers < 2)
    with pytest.raises(ValueError, match="num_layers must be at least 2"):
        GATClassifier(GATConfig(num_layers=1))


def test_stratified_node_split():
    """Verify stratified partitioning into train_mask and val_mask without overlap."""
    # 50 nodes, exactly 10 nodes per class across 5 classes
    y = torch.tensor([c for c in range(5) for _ in range(10)], dtype=torch.long)
    train_mask, val_mask = create_stratified_node_split(y, val_ratio=0.2, random_seed=42)

    assert train_mask.shape == (50,)
    assert val_mask.shape == (50,)
    # No overlap
    assert not (train_mask & val_mask).any()
    # Total covered
    assert (train_mask | val_mask).all()

    # Stratification check (8 train, 2 val per class)
    for c in range(5):
        c_mask = (y == c)
        assert (train_mask & c_mask).sum().item() == 8
        assert (val_mask & c_mask).sum().item() == 2


def test_short_training_run_on_micro_graph(tmp_path):
    """Verify that train_gat runs end-to-end on a small graph and generates checkpoints."""
    # Create small synthetic graph
    torch.manual_seed(99)
    x = torch.randn(30, 45, dtype=torch.float32)
    src = list(range(30)) + list(range(0, 30, 2))
    dst = list(range(1, 30)) + [0] + list(range(1, 30, 2))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor([i % 5 for i in range(30)], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=30)

    graph_file = str(tmp_path / "micro_graph.pt")
    torch.save(data, graph_file)

    config = GATConfig(
        in_dim=45,
        hidden_dim=16,
        num_classes=5,
        heads=2,
        epochs=4,
        learning_rate=0.01
    )

    out_dir = str(tmp_path / "models")
    hist_dir = str(tmp_path / "results")

    results = train_gat(
        train_graph_path=graph_file,
        config=config,
        output_dir=out_dir,
        history_dir=hist_dir,
        log_interval=2
    )

    assert os.path.exists(results["checkpoint_path"])
    assert os.path.exists(results["history_path"])
    assert results["final_epoch"] == 4
    assert results["best_val_macro_f1"] >= 0.0
