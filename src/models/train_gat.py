"""
Training Pipeline for GAT-XIDS
==============================
Trains the Graph Attention Network for IoMT intrusion detection on flow-similarity graphs.

METHODOLOGICAL GUARANTEES:
--------------------------
1. Transductive Validation Masking:
   In graph node classification, a single connected graph is used for message passing.
   However, node indices are partitioned into train_mask and val_mask.
   Only nodes in train_mask contribute to loss calculation and gradient backpropagation.
   Validation nodes receive graph messages from neighbors, but their labels are strictly
   withheld during optimization and used exclusively for early stopping and model selection.

2. Strict Held-Out Test Isolation:
   The held-out testing graph (e.g. test_sample_graph.pt) is NOT loaded or accessed during training,
   early stopping, or hyperparameter selection. It serves strictly as an independent evaluation benchmark.

3. Class-Weighted CrossEntropyLoss:
   To counter IoMT attack class imbalances, loss penalties are weighted inversely proportional to
   class frequencies computed exclusively on the training subset (train_mask).
"""

import os
import json
import time
import logging
from dataclasses import asdict
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import torch
import torch.nn as nn
import sklearn.metrics as metrics
from torch_geometric.data import Data

from src.models.gat_model import GATClassifier, GATConfig
from src.data.preprocess import ID_TO_CATEGORY, CATEGORY_TO_ID

logger = logging.getLogger("gat_xids.train_gat")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def create_stratified_node_split(
    y: torch.Tensor,
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Splits node indices into boolean train_mask and val_mask with stratified class representation.
    
    Returns:
        (train_mask, val_mask): Tuple of boolean tensors of shape [N].
    """
    n_nodes = y.shape[0]
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)

    rng = np.random.RandomState(random_seed)
    y_np = y.cpu().numpy()
    unique_classes = np.unique(y_np)

    for cls in unique_classes:
        cls_indices = np.where(y_np == cls)[0]
        rng.shuffle(cls_indices)
        n_cls = len(cls_indices)
        
        # Determine validation size for class
        n_val = max(1, int(round(val_ratio * n_cls))) if n_cls >= 2 else 0
        n_train = n_cls - n_val

        val_indices = cls_indices[:n_val]
        train_indices = cls_indices[n_val:]

        val_mask[val_indices] = True
        train_mask[train_indices] = True

    return train_mask, val_mask


def compute_class_weights(y: torch.Tensor, mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Computes inverse class frequencies on the training mask only to balance CrossEntropyLoss.
    Formula: weight_c = N_train / (num_classes * N_c)
    """
    y_masked = y[mask]
    counts = torch.bincount(y_masked, minlength=num_classes).float()
    total = float(len(y_masked))

    weights = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (float(num_classes) * counts[c])
        else:
            weights[c] = 1.0

    return weights


def evaluate_metrics(
    model: GATClassifier,
    data: Data,
    mask: torch.Tensor,
    criterion: nn.Module
) -> Dict[str, float]:
    """
    Evaluates loss, accuracy, and macro precision/recall/F1 on a specified node mask.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, return_attention_weights=False)
        loss = criterion(logits[mask], data.y[mask]).item()

        preds = torch.argmax(logits[mask], dim=-1).cpu().numpy()
        targets = data.y[mask].cpu().numpy()

        acc = float(metrics.accuracy_score(targets, preds))
        f1_macro = float(metrics.f1_score(targets, preds, average="macro", zero_division=0))
        precision_macro = float(metrics.precision_score(targets, preds, average="macro", zero_division=0))
        recall_macro = float(metrics.recall_score(targets, preds, average="macro", zero_division=0))

    return {
        "loss": round(loss, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(f1_macro, 4),
        "macro_precision": round(precision_macro, 4),
        "macro_recall": round(recall_macro, 4)
    }


def train_gat(
    train_graph_path: str,
    config: Optional[GATConfig] = None,
    output_dir: str = "models",
    history_dir: str = "results",
    log_interval: int = 5
) -> Dict[str, Any]:
    """
    Executes training of GATClassifier on the supplied training graph.
    """
    if config is None:
        config = GATConfig()

    # Reproducibility seeds
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    if not os.path.exists(train_graph_path):
        raise FileNotFoundError(f"Training graph not found: {train_graph_path}")

    logger.info(f"Loading training graph: {train_graph_path}")
    data = torch.load(train_graph_path, weights_only=False)

    num_nodes = data.num_nodes
    in_dim = data.x.shape[1]
    num_classes = len(torch.unique(data.y))

    logger.info(
        f"Graph loaded: {num_nodes} nodes, {data.edge_index.shape[1]} edges, "
        f"{in_dim} features, {num_classes} unique classes."
    )

    # Reconcile config dimensions with graph data
    config.in_dim = in_dim
    config.num_classes = num_classes

    # Stratified Train/Val Split on Training Graph
    train_mask, val_mask = create_stratified_node_split(
        data.y,
        val_ratio=config.val_ratio,
        random_seed=config.random_seed
    )
    n_train = int(train_mask.sum())
    n_val = int(val_mask.sum())
    logger.info(f"Node Split: {n_train} training nodes ({100*n_train/num_nodes:.1f}%), {n_val} validation nodes ({100*n_val/num_nodes:.1f}%)")

    # Compute class weights exclusively on training nodes
    class_weights = compute_class_weights(data.y, train_mask, num_classes)
    logger.info(f"Computed training class weights: {class_weights.tolist()}")

    # Initialize model, loss, optimizer
    model = GATClassifier(config)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = os.path.join(output_dir, "gat_best_checkpoint.pt")

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": []
    }

    start_time = time.time()
    logger.info(f"Starting GAT training for up to {config.epochs} epochs (Early stopping patience: {config.early_stopping_patience})...")

    for epoch in range(1, config.epochs + 1):
        # 1. Training step
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index, return_attention_weights=False)
        loss = criterion(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Compute training accuracy
        with torch.no_grad():
            train_preds = torch.argmax(logits[train_mask], dim=-1)
            train_acc = float((train_preds == data.y[train_mask]).float().mean().item())

        # 2. Validation step
        val_metrics = evaluate_metrics(model, data, val_mask, criterion)

        history["train_loss"].append(round(loss.item(), 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        if epoch % log_interval == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:03d}/{config.epochs:03d} | "
                f"Train Loss: {loss.item():.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['macro_f1']:.4f}"
            )

        # 3. Model checkpointing on validation macro F1
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0

            extra_meta = {
                "best_epoch": best_epoch,
                "best_val_metrics": val_metrics,
                "class_weights": class_weights.tolist(),
                "num_classes": num_classes,
                "in_dim": in_dim
            }
            model.save_checkpoint(best_checkpoint_path, extra_metadata=extra_meta)
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (no val F1 improvement for {config.early_stopping_patience} epochs).")
                break

    total_training_time = round(time.time() - start_time, 2)
    logger.info(f"Training completed in {total_training_time}s. Best epoch: {best_epoch} with Val Macro F1: {best_val_f1:.4f}")

    # Save training history JSON
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({
            "config": asdict(config),
            "training_time_seconds": total_training_time,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "history": history
        }, f, indent=2)

    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "total_training_time": total_training_time,
        "checkpoint_path": best_checkpoint_path,
        "history_path": history_path,
        "final_epoch": len(history["train_loss"]),
        "train_nodes": n_train,
        "val_nodes": n_val
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GAT-XIDS Training Pipeline")
    parser.add_argument("--train-graph", type=str, default="results/graphs/train_sample_graph.pt")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 0.005)")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension (default: 32)")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads (default: 4)")
    parser.add_argument("--output-dir", type=str, default="models", help="Checkpoint directory")
    parser.add_argument("--history-dir", type=str, default="results", help="History directory")

    args = parser.parse_args()

    config = GATConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        heads=args.heads
    )

    results = train_gat(
        train_graph_path=args.train_graph,
        config=config,
        output_dir=args.output_dir,
        history_dir=args.history_dir
    )

    print("\n" + "="*60)
    print("GAT TRAINING EXPERIMENT RESULTS")
    print("="*60)
    print(f"Training Nodes     : {results['train_nodes']} | Validation Nodes: {results['val_nodes']}")
    print(f"Total Epochs Run   : {results['final_epoch']} (Best Epoch: {results['best_epoch']})")
    print(f"Best Val Macro F1  : {results['best_val_macro_f1']:.4f}")
    print(f"Training Duration  : {results['total_training_time']}s")
    print(f"Saved Checkpoint   : {results['checkpoint_path']}")
    print(f"Training History   : {results['history_path']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
