import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler


class MiniGAT(nn.Module):
    """Small 2-layer Graph Attention Network for smoke testing GAT-XIDS."""
    def __init__(self, in_features, hidden_dim, num_classes, heads=2):
        super().__init__()
        self.conv1 = GATConv(in_features, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, num_classes, heads=1)

    def forward(self, x, edge_index, return_attention=False):
        if return_attention:
            x, (edge_idx1, alpha1) = self.conv1(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            out, (edge_idx2, alpha2) = self.conv2(x, edge_index, return_attention_weights=True)
            return out, (edge_idx1, alpha1), (edge_idx2, alpha2)
        else:
            x = F.elu(self.conv1(x, edge_index))
            out = self.conv2(x, edge_index)
            return out


def test_mini_gat_smoke():
    """Verify that a small flow feature batch can be converted to graph and forward-passed."""
    # 1. Sample 5 rows from an existing dataset CSV
    sample_csv = os.path.join("WIFI & MQTT", "attacks", "CSV", "train", "Benign_train.pcap.csv")
    if not os.path.exists(sample_csv):
        sample_csv = "sample_input.csv"
    assert os.path.exists(sample_csv), f"Sample CSV not found at {sample_csv}"
    
    df_sample = pd.read_csv(sample_csv, nrows=5)
    assert len(df_sample) >= 5, f"Sample CSV must contain at least 5 rows, got {len(df_sample)}"
    assert df_sample.shape[1] == 45, f"Sample CSV must contain exactly 45 feature columns, got {df_sample.shape[1]}"
    df_sample = df_sample.iloc[:5]
    assert df_sample.shape == (5, 45)

    # 2. Scale features
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(df_sample.values)
    x = torch.tensor(scaled_feats, dtype=torch.float32)

    # 3. Construct a toy edge index (e.g. fully connected directed graph between the 5 sample flows)
    num_nodes = 5
    source_nodes = []
    target_nodes = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                source_nodes.append(i)
                target_nodes.append(j)
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    # 4. Instantiate MiniGAT
    model = MiniGAT(in_features=45, hidden_dim=16, num_classes=2, heads=2)
    model.eval()

    # 5. Forward pass with explainability attention weights
    with torch.no_grad():
        out, att1, att2 = model(x, edge_index, return_attention=True)

    assert out.shape == (5, 2)
    # att1[0] is edge_index with self-loops, att1[1] is attention weights
    assert att1[1].shape[1] == 2  # 2 heads
    assert att2[1].shape[1] == 1  # 1 head
