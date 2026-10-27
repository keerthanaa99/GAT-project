"""
Graph Attention Network (GAT) Architecture for GAT-XIDS
=======================================================
Implements an explainable Graph Attention Network (GAT) for intrusion detection in IoMT networks.
Provides multi-head attention over flow-similarity graphs and structured extraction of attention
coefficients for interpretability and forensic auditing.

ARCHITECTURAL PRINCIPLES:
-------------------------
1. Self-Attention Formulation (Veličković et al., ICLR 2018):
   For connected nodes i and j, the attention coefficient alpha_ij quantifies the importance of node j's
   flow characteristics to node i's traffic classification:
       alpha_ij = Softmax_j( LeakyReLU( a^T [ W h_i || W h_j ] ) )

2. Multi-Head Attention:
   K independent attention mechanisms stabilize attention learning and expand representation capacity.
   In intermediate layers, multi-head features are concatenated: h'_i = ||_{k=1}^K sigma( sum_j alpha_ij^k W^k h_j ).
   In the final layer, heads are projected to class logits: z_i = sum_j alpha_ij W h_j.

3. Explainability Integration:
   Each GATConv layer natively returns (edge_index, alpha) when return_attention_weights=True.
   These attention tensors are tracked per layer to enable downstream visualization and attribution.
   Note: High attention indicates that the network relied on neighbor j's feature representations
   during message passing; it is a diagnostic metric, not a mathematical proof of causal maliciousness.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from src.data.preprocess import EXPECTED_FEATURES, CATEGORY_TO_ID, ID_TO_CATEGORY


@dataclass
class GATConfig:
    """Hyperparameters and architectural configuration for GATClassifier."""
    in_dim: int = 45                  # Number of input numerical features
    hidden_dim: int = 32              # Hidden feature dimension per attention head
    num_classes: int = 5              # Number of output classes (5 categories or 19 subtypes or 2 binary)
    heads: int = 4                    # Number of attention heads in hidden layers
    num_layers: int = 2               # Total GAT layers (minimum 2: input->hidden, hidden->output)
    dropout: float = 0.2              # Dropout probability for features and attention weights
    negative_slope: float = 0.2       # LeakyReLU negative slope for attention scoring
    add_self_loops: bool = True       # Whether GATConv adds self-loops to preserve central node state
    target_label: str = "category_id" # Label field in Data object
    learning_rate: float = 0.005      # Adam optimizer learning rate
    weight_decay: float = 5e-4        # L2 regularization factor
    epochs: int = 50                  # Max training epochs
    val_ratio: float = 0.2            # Proportion of training graph nodes reserved for validation
    early_stopping_patience: int = 15 # Patience for early stopping on validation loss
    random_seed: int = 42             # Seed for reproducible initialization and splitting


class GATClassifier(nn.Module):
    """
    Configurable Graph Attention Network for node classification and intrusion detection.
    """
    def __init__(self, config: Optional[GATConfig] = None):
        super().__init__()
        if config is None:
            config = GATConfig()
        self.config = config

        if config.num_layers < 2:
            raise ValueError(f"num_layers must be at least 2, got {config.num_layers}")
        if config.in_dim <= 0 or config.num_classes <= 0:
            raise ValueError(f"in_dim and num_classes must be positive integers, got {config.in_dim}, {config.num_classes}")
        if config.heads <= 0 or config.hidden_dim <= 0:
            raise ValueError(f"heads and hidden_dim must be positive integers, got {config.heads}, {config.hidden_dim}")

        self.layers = nn.ModuleList()

        # Input layer: [in_dim] -> [hidden_dim * heads]
        self.layers.append(
            GATConv(
                in_channels=config.in_dim,
                out_channels=config.hidden_dim,
                heads=config.heads,
                concat=True,
                negative_slope=config.negative_slope,
                dropout=config.dropout,
                add_self_loops=config.add_self_loops
            )
        )

        # Intermediate hidden layers (if num_layers > 2)
        for _ in range(config.num_layers - 2):
            self.layers.append(
                GATConv(
                    in_channels=config.hidden_dim * config.heads,
                    out_channels=config.hidden_dim,
                    heads=config.heads,
                    concat=True,
                    negative_slope=config.negative_slope,
                    dropout=config.dropout,
                    add_self_loops=config.add_self_loops
                )
            )

        # Output classification layer: [hidden_dim * heads] -> [num_classes]
        self.layers.append(
            GATConv(
                in_channels=config.hidden_dim * config.heads,
                out_channels=config.num_classes,
                heads=1,
                concat=False,
                negative_slope=config.negative_slope,
                dropout=config.dropout,
                add_self_loops=config.add_self_loops
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass through GAT layers.

        Args:
            x: Node feature matrix of shape [N, in_dim]
            edge_index: Graph connectivity tensor of shape [2, E]
            return_attention_weights: If True, returns list of (edge_index_l, alpha_l) per layer.

        Returns:
            logits: Tensor of shape [N, num_classes]
            attentions (optional): List of tuples [(edge_idx_1, alpha_1), ...]
        """
        if x.dim() != 2 or x.shape[1] != self.config.in_dim:
            raise ValueError(f"Expected x of shape [N, {self.config.in_dim}], got {x.shape}")
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Expected edge_index of shape [2, E], got {edge_index.shape}")
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input feature tensor contains NaN or Inf values")

        attentions: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Forward pass through hidden layers with ELU and dropout
        for layer in self.layers[:-1]:
            if return_attention_weights:
                x, (edge_idx, alpha) = layer(x, edge_index, return_attention_weights=True)
                attentions.append((edge_idx, alpha))
            else:
                x = layer(x, edge_index)

            x = F.elu(x)
            x = F.dropout(x, p=self.config.dropout, training=self.training)

        # Final classification layer (raw logits)
        final_layer = self.layers[-1]
        if return_attention_weights:
            logits, (edge_idx, alpha) = final_layer(x, edge_index, return_attention_weights=True)
            attentions.append((edge_idx, alpha))
            return logits, attentions
        else:
            logits = final_layer(x, edge_index)
            return logits

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Dict[str, Any]:
        """
        Inference method returning logits, class probabilities (softmax), predicted classes,
        and optional attention weights.
        """
        self.eval()
        if return_attention_weights:
            logits, attentions = self.forward(x, edge_index, return_attention_weights=True)
        else:
            logits = self.forward(x, edge_index, return_attention_weights=False)
            attentions = None

        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        result = {
            "logits": logits,
            "probabilities": probs,
            "predicted_classes": preds,
        }
        if attentions is not None:
            result["attentions"] = attentions

        return result

    def save_checkpoint(
        self,
        filepath: str,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Saves model weights, config, and feature schema to a PyTorch checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": asdict(self.config),
            "expected_features": EXPECTED_FEATURES,
            "extra_metadata": extra_metadata or {}
        }
        torch.save(checkpoint, filepath)
        return filepath

    @classmethod
    def from_checkpoint(cls, filepath: str) -> "GATClassifier":
        """Loads a GATClassifier from a saved checkpoint file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

        checkpoint = torch.load(filepath, weights_only=False)
        config = GATConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
