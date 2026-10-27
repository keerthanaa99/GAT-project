import os
import json
import pytest
import numpy as np
import torch
from torch_geometric.data import Data

from src.models.gat_model import GATClassifier, GATConfig
from src.explain.attention_explainer import AttentionExplainer
from src.explain.explanation_report import format_explanation_report, save_explanation_artifacts


@pytest.fixture
def test_graph():
    """Loads the real held-out test graph artifact."""
    graph_path = "results/graphs/test_sample_graph.pt"
    assert os.path.exists(graph_path), f"Test graph artifact missing at {graph_path}"
    return torch.load(graph_path, weights_only=False)


@pytest.fixture
def explainer():
    """Initializes explainer using the saved best GAT checkpoint."""
    checkpoint_path = "models/gat_best_checkpoint.pt"
    assert os.path.exists(checkpoint_path), f"Checkpoint missing at {checkpoint_path}"
    return AttentionExplainer(checkpoint_path=checkpoint_path)


def test_checkpoint_and_graph_loading(explainer):
    """Verify that checkpoint and model are properly loaded in eval mode."""
    assert explainer.model is not None
    assert not explainer.model.training  # Must be in evaluation mode
    assert explainer.config.in_dim == 45
    assert explainer.config.num_classes == 5

    with pytest.raises(FileNotFoundError):
        AttentionExplainer(checkpoint_path="non_existent_checkpoint.pt")


def test_valid_target_node_explanation(explainer, test_graph):
    """Verify explanation structure, prediction validity, and probability ranges for node 0."""
    exp = explainer.explain_node(test_graph, node_index=0, top_k=5)

    assert "target_node" in exp
    assert "prediction" in exp
    assert "self_attention" in exp
    assert "top_contributing_neighbors" in exp
    assert "scientific_disclaimer" in exp

    # Target node verification
    target = exp["target_node"]
    assert target["node_index"] == 0
    assert target["true_category"] == test_graph.label_category[0]

    # Prediction verification
    pred = exp["prediction"]
    assert 0 <= pred["predicted_class_id"] < 5
    assert 0.0 <= pred["confidence"] <= 1.0
    assert len(pred["class_probabilities"]) == 5
    assert np.isclose(sum(pred["class_probabilities"].values()), 1.0, atol=1e-5)


def test_invalid_node_index_and_top_k(explainer, test_graph):
    """Verify proper exception handling for negative, out-of-bounds, or invalid top_k."""
    n = test_graph.num_nodes

    with pytest.raises(IndexError, match="Invalid node_index"):
        explainer.explain_node(test_graph, node_index=-1)

    with pytest.raises(IndexError, match="Invalid node_index"):
        explainer.explain_node(test_graph, node_index=n)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        explainer.explain_node(test_graph, node_index=0, top_k=0)


def test_attention_dimensions_and_sum(explainer, test_graph):
    """Verify that multi-head attention weights sum to 1.0 across incoming edges (including self-loop)."""
    node_idx = 10
    exp = explainer.explain_node(test_graph, node_index=node_idx, top_k=5)

    all_nbrs = exp["all_neighbors_detail"]
    assert len(all_nbrs) > 0

    # Layer 2 decision attention must sum to 1.0 across all incoming edges (including self)
    l2_sum = sum(nbr["layer_2_attention"] for nbr in all_nbrs)
    assert np.isclose(l2_sum, 1.0, atol=1e-4)

    # Layer 1 mean attention must also sum to 1.0
    l1_sum = sum(nbr["layer_1_mean_attention"] for nbr in all_nbrs)
    assert np.isclose(l1_sum, 1.0, atol=1e-4)


def test_correct_top_neighbor_ranking(explainer, test_graph):
    """Verify that external neighbors are ranked monotonically by attention weight and exclude self-loops."""
    node_idx = 0
    exp = explainer.explain_node(test_graph, node_index=node_idx, top_k=5)
    top_nbrs = exp["top_contributing_neighbors"]

    assert len(top_nbrs) <= 5
    for rank, nbr in enumerate(top_nbrs):
        # Must not contain self
        assert nbr["neighbor_index"] != node_idx
        assert not nbr["is_self"]

        # Check descending order
        if rank > 0:
            prev = top_nbrs[rank - 1]
            assert prev["layer_2_attention"] >= nbr["layer_2_attention"]


def test_multi_head_attention_handling(explainer, test_graph):
    """Verify that multi-head attention weights are preserved per-head and match the mean."""
    exp = explainer.explain_node(test_graph, node_index=0, top_k=3)

    for nbr in exp["top_contributing_neighbors"]:
        head_alphas = nbr["layer_1_head_attentions"]
        assert len(head_alphas) == 4  # 4 heads in layer 1
        expected_mean = float(np.mean(head_alphas))
        assert np.isclose(nbr["layer_1_mean_attention"], expected_mean, atol=1e-5)


def test_metadata_alignment(explainer, test_graph):
    """Verify that neighbor and target metadata matches ground-truth graph attributes."""
    exp = explainer.explain_node(test_graph, node_index=5, top_k=3)

    target_meta = exp["target_node"]
    assert target_meta["subtype"] == test_graph.label_subtype[5]
    assert target_meta["source_file"] == test_graph.label_raw[5]

    for nbr in exp["top_contributing_neighbors"]:
        nbr_idx = nbr["neighbor_index"]
        assert nbr["metadata"]["category"] == test_graph.label_category[nbr_idx]
        assert nbr["metadata"]["subtype"] == test_graph.label_subtype[nbr_idx]


def test_repeatable_output_evaluation_mode(explainer, test_graph):
    """Verify that running explain_node twice on the same node produces perfectly deterministic results."""
    exp1 = explainer.explain_node(test_graph, node_index=15, top_k=5)
    exp2 = explainer.explain_node(test_graph, node_index=15, top_k=5)

    assert exp1["prediction"] == exp2["prediction"]
    assert exp1["self_attention"] == exp2["self_attention"]
    assert len(exp1["top_contributing_neighbors"]) == len(exp2["top_contributing_neighbors"])

    for n1, n2 in zip(exp1["top_contributing_neighbors"], exp2["top_contributing_neighbors"]):
        assert n1["neighbor_index"] == n2["neighbor_index"]
        assert n1["layer_2_attention"] == n2["layer_2_attention"]


def test_json_and_txt_serialization(tmp_path, explainer, test_graph):
    """Verify JSON and formatted text artifact generation and readability."""
    exp = explainer.explain_node(test_graph, node_index=0, top_k=5)
    out_dir = str(tmp_path / "explanations")

    json_path, txt_path = save_explanation_artifacts(exp, out_dir, prefix="node_0_test")

    assert os.path.exists(json_path)
    assert os.path.exists(txt_path)

    # Validate JSON deserialization
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["target_node"]["node_index"] == 0
    assert loaded["prediction"]["predicted_category"] == exp["prediction"]["predicted_category"]

    # Validate text report contents
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    assert "GAT-XIDS INHERENT ATTENTION EXPLANATION REPORT" in text
    assert "TARGET FLOW NODE OVERVIEW" in text
    assert "TOP-5 CONTRIBUTING NEIGHBOR FLOWS" in text
    assert "SCIENTIFIC & FORENSIC DISCLAIMER" in text


def test_missing_metadata_handling(explainer):
    """Verify that explanation works cleanly on un-annotated graphs lacking y or metadata attributes."""
    # Graph with strictly x and edge_index (unlabeled prediction)
    x = torch.randn(10, 45, dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    bare_data = Data(x=x, edge_index=edge_index, num_nodes=10)

    exp = explainer.explain_node(bare_data, node_index=1, top_k=2)

    assert exp["target_node"]["true_category"] is None
    assert exp["target_node"]["source_file"] is None
    assert exp["prediction"]["predicted_category"] is not None
    assert len(exp["top_contributing_neighbors"]) > 0
