import sys
import pytest
import numpy as np
import pandas as pd
import sklearn
import networkx as nx
import matplotlib
import streamlit
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


def test_python_version():
    """Verify that Python version is 3.11 or 3.12."""
    major = sys.version_info.major
    minor = sys.version_info.minor
    assert major == 3
    assert minor in (11, 12), f"Expected Python 3.11 or 3.12, got {major}.{minor}"


def test_torch_cpu_tensor_ops():
    """Verify PyTorch imports, runs on CPU, and performs tensor math."""
    assert torch.__version__.startswith("2.")
    assert not torch.cuda.is_available()  # Ensure CPU environment
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    y = torch.tensor([[2.0, 0.0], [1.0, 3.0]], dtype=torch.float32)
    z = torch.matmul(x, y)
    expected = torch.tensor([[4.0, 6.0], [10.0, 12.0]], dtype=torch.float32)
    assert torch.allclose(z, expected)


def test_torch_geometric_gatconv():
    """Verify PyTorch Geometric imports and GATConv runs forward pass with attention weights."""
    in_channels = 8
    out_channels = 4
    heads = 2

    # Instantiate GATConv
    conv = GATConv(in_channels=in_channels, out_channels=out_channels, heads=heads)

    # Create a small synthetic graph: 4 nodes, 4 edges
    x = torch.randn(4, in_channels)
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 0]
    ], dtype=torch.long)

    # Forward pass returning attention weights (critical for GAT-XIDS explainability)
    out, (edge_idx, alpha) = conv(x, edge_index, return_attention_weights=True)

    assert out.shape == (4, out_channels * heads)
    # By default, GATConv adds self-loops (4 input edges + 4 self-loops = 8 edges)
    assert edge_idx.shape == (2, 8)
    assert alpha.shape == (8, heads)


def test_ecosystem_imports():
    """Verify data science and utility dependencies."""
    assert np.__version__ is not None
    assert pd.__version__ is not None
    assert sklearn.__version__ is not None
    assert nx.__version__ is not None
    assert matplotlib.__version__ is not None
    assert streamlit.__version__ is not None
    assert pytest.__version__ is not None


if __name__ == "__main__":
    test_python_version()
    test_torch_cpu_tensor_ops()
    test_torch_geometric_gatconv()
    test_ecosystem_imports()
    print("All environment and smoke tests passed successfully!")
