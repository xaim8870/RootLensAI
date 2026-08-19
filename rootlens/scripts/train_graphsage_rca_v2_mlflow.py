#scripts/train_graphsage_rca_v2_mlflow.py
#!/usr/bin/env python3
"""
Train RootLens GraphSAGE RCA v2 with node-identity-preserving readout.

Why v2?
-------
GraphSAGE v1 used global_mean_pool, which averages all service-node embeddings.
For root-cause localization this can discard WHICH service produced a signal.

v2 keeps the frozen 12-service node order and concatenates all learned node
embeddings before the classifier:

    12 nodes x 7 features
        -> SAGEConv(7 -> hidden)
        -> BatchNorm + ReLU + Dropout
        -> SAGEConv(hidden -> hidden)
        -> BatchNorm + ReLU
        -> reshape [batch, 12 * hidden]
        -> MLP
        -> 5 RCA classes

This preserves:
    1. graph message passing
    2. service identity
    3. cross-service context

Research policy:
- train on train_graphs.pt only
- validation used for early stopping
- test remains sealed
- normalization fitted on train graphs only
- tracked with MLflow
- visual comparisons generated against previous models

Usage:
    python rootlens\\scripts\\train_graphsage_rca_v2_mlflow.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, SAGEConv


EXPERIMENT_NAME = "RootLens-RCA-GNN-v2"

CLASS_TO_INDEX = {
    "healthy": 0,
    "payment": 1,
    "cart": 2,
    "checkout": 3,
    "product_catalog": 4,
}
INDEX_TO_CLASS = {v: k for k, v in CLASS_TO_INDEX.items()}
CLASS_ORDER = ["healthy", "payment", "cart", "checkout", "product_catalog"]

NODE_FEATURES = [
    "cpu",
    "memory",
    "request_rate",
    "has_requests",
    "latency_ms",
    "error_rps",
    "error_rate",
]

NODE_COUNT = 12
NODE_FEATURE_COUNT = 7


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(path: Path, repo_root: Path) -> Path:
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_graphs(path: Path) -> list[Data]:
    graphs = torch.load(path, weights_only=False)
    if not isinstance(graphs, list) or not graphs:
        raise ValueError(f"Invalid graph list: {path}")
    return graphs


def validate_graphs(
    graphs: list[Data],
    split_name: str,
    expected_graphs: int,
    expected_runs: int,
) -> None:
    if len(graphs) != expected_graphs:
        raise ValueError(
            f"{split_name}: expected {expected_graphs} graphs, found {len(graphs)}"
        )

    runs = set()
    labels = set()
    ref_edge_index = graphs[0].edge_index

    for i, g in enumerate(graphs):
        if tuple(g.x.shape) != (NODE_COUNT, NODE_FEATURE_COUNT):
            raise ValueError(
                f"{split_name} graph {i}: expected x=(12,7), got {tuple(g.x.shape)}"
            )
        if torch.isnan(g.x).any() or torch.isinf(g.x).any():
            raise ValueError(f"{split_name} graph {i}: invalid feature values")
        if not torch.equal(g.edge_index, ref_edge_index):
            raise ValueError(f"{split_name} graph {i}: topology mismatch")

        y = int(g.y.item())
        labels.add(INDEX_TO_CLASS[y])
        runs.add(str(g.run_id))

    if len(runs) != expected_runs:
        raise ValueError(
            f"{split_name}: expected {expected_runs} runs, found {len(runs)}"
        )
    if labels != set(CLASS_ORDER):
        raise ValueError(
            f"{split_name}: class coverage mismatch: {sorted(labels)}"
        )


def check_run_leakage(train_graphs: list[Data], val_graphs: list[Data]) -> None:
    train_runs = {str(g.run_id) for g in train_graphs}
    val_runs = {str(g.run_id) for g in val_graphs}
    overlap = train_runs & val_runs
    if overlap:
        raise ValueError(f"RUN LEAKAGE: {sorted(overlap)}")


def compute_normalization(
    train_graphs: list[Data],
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.cat([g.x.float() for g in train_graphs], dim=0)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return mean, std


def normalize_graphs(
    graphs: list[Data],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> list[Data]:
    out = []
    for graph in graphs:
        g = copy.deepcopy(graph)
        g.x = (g.x.float() - mean) / std
        out.append(g)
    return out


class RootLensGraphSAGEV2(nn.Module):
    """
    Node-identity-preserving GraphSAGE classifier.

    Critical assumption:
        Every graph has exactly 12 nodes in the same frozen node order.
    """

    def __init__(
        self,
        in_channels: int = 7,
        hidden_channels: int = 48,
        num_classes: int = 5,
        dropout: float = 0.30,
        mlp_hidden: int = 96,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels
        self.node_count = NODE_COUNT
        self.dropout_p = dropout

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)

        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)

        self.classifier = nn.Sequential(
            nn.Linear(NODE_COUNT * hidden_channels, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout_p, training=self.training)

        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        # Small residual connection.
        x = x1 + x2

        # PyG batches graphs by concatenating nodes. Because every graph has
        # exactly 12 nodes in the same order, reshape preserves service identity.
        batch_size = int(batch.max().item()) + 1
        expected_nodes = batch_size * NODE_COUNT

        if x.shape[0] != expected_nodes:
            raise RuntimeError(
                f"Expected {expected_nodes} batched nodes, found {x.shape[0]}"
            )

        x = x.view(batch_size, NODE_COUNT * self.hidden_channels)
        return self.classifier(x)


def class_weights(
    graphs: list[Data],
    device: torch.device,
) -> torch.Tensor:
    counts = np.zeros(len(CLASS_ORDER), dtype=np.int64)
    for g in graphs:
        counts[int(g.y.item())] += 1

    total = counts.sum()
    weights = total / (len(counts) * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(batch.x, batch.edge_index, batch.batch)
        targets = batch.y.view(-1)

        loss = criterion(logits, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        n = targets.shape[0]
        total_loss += float(loss.item()) * n
        total += n

    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total = 0

    y_true_idx: list[int] = []
    y_pred_idx: list[int] = []
    confidences: list[float] = []
    run_ids: list[str] = []
    timestamps: list[str] = []

    for batch in loader:
        batch = batch.to(device)

        logits = model(batch.x, batch.edge_index, batch.batch)
        targets = batch.y.view(-1)
        loss = criterion(logits, targets)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        conf = probs.max(dim=1).values

        n = targets.shape[0]
        total_loss += float(loss.item()) * n
        total += n

        y_true_idx.extend(targets.cpu().tolist())
        y_pred_idx.extend(preds.cpu().tolist())
        confidences.extend(conf.cpu().tolist())
        run_ids.extend([str(v) for v in batch.run_id])
        timestamps.extend([str(v) for v in batch.timestamp])

    y_true = [INDEX_TO_CLASS[i] for i in y_true_idx]
    y_pred = [INDEX_TO_CLASS[i] for i in y_pred_idx]

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, labels=CLASS_ORDER, average="weighted", zero_division=0
        ),
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": confidences,
        "run_ids": run_ids,
        "timestamps": timestamps,
    }


def save_loss_plot(history: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss")
    ax.plot(history["epoch"], history["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("RootLens GraphSAGE RCA v2 — Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_metric_plot(history: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(history["epoch"], history["val_accuracy"], label="Validation accuracy")
    ax.plot(history["epoch"], history["val_macro_f1"], label="Validation macro F1")
    ax.plot(
        history["epoch"],
        history["val_weighted_f1"],
        label="Validation weighted F1",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title("RootLens GraphSAGE RCA v2 — Validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_confusion(y_true: list[str], y_pred: list[str], path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    display = ConfusionMatrixDisplay(cm, display_labels=CLASS_ORDER)

    fig, ax = plt.subplots(figsize=(9, 7))
    display.plot(
        ax=ax,
        xticks_rotation=35,
        values_format="d",
        colorbar=False,
    )
    ax.set_title("RootLens GraphSAGE RCA v2 — Validation Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_model_comparison(
    repo_root: Path,
    v2_metrics: dict[str, float],
    path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    # Prefer actual baseline CSV if available.
    baseline_csv = (
        repo_root
        / "rootlens/data/reports/baseline_results_v1/baseline_comparison.csv"
    )

    if baseline_csv.is_file():
        df = pd.read_csv(baseline_csv)
        for _, r in df.iterrows():
            rows.append(
                {
                    "model": str(r["model"]),
                    "val_accuracy": float(r["val_accuracy"]),
                    "val_macro_f1": float(r["val_macro_f1"]),
                    "val_weighted_f1": float(r["val_weighted_f1"]),
                }
            )

    # Prefer actual v1 GNN summary if available.
    v1_summary_path = (
        repo_root
        / "rootlens/data/reports/graphsage_v1/run_summary.json"
    )

    if v1_summary_path.is_file():
        v1 = load_json(v1_summary_path)
        rows.append(
            {
                "model": "graphsage_v1_mean_pool",
                "val_accuracy": float(v1["best_validation_accuracy"]),
                "val_macro_f1": float(v1["best_validation_macro_f1"]),
                "val_weighted_f1": float(v1["best_validation_weighted_f1"]),
            }
        )

    rows.append(
        {
            "model": "graphsage_rca_v2_node_preserving",
            "val_accuracy": v2_metrics["accuracy"],
            "val_macro_f1": v2_metrics["macro_f1"],
            "val_weighted_f1": v2_metrics["weighted_f1"],
        }
    )

    comparison = pd.DataFrame(rows).drop_duplicates(
        subset=["model"], keep="last"
    )
    comparison = comparison.sort_values("val_macro_f1", ascending=False)

    csv_path = path.with_suffix(".csv")
    comparison.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = comparison.sort_values("val_macro_f1", ascending=True)
    ax.barh(plot_df["model"], plot_df["val_macro_f1"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Validation Macro F1")
    ax.set_title("RootLens Model Comparison — Validation Macro F1")

    for y, value in enumerate(plot_df["val_macro_f1"]):
        ax.text(
            min(value + 0.01, 0.98),
            y,
            f"{value:.4f}",
            va="center",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return comparison


def configure_mlflow(
    repo_root: Path,
    experiment_name: str = EXPERIMENT_NAME,
) -> tuple[str, Path]:
    db_path = (repo_root / "rootlens/mlflow.db").resolve()
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri, db_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train node-preserving RootLens GraphSAGE RCA v2."
    )

    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("rootlens/data/processed/dataset_v1/graphs"),
    )
    p.add_argument(
        "--dataset-metadata",
        type=Path,
        default=Path(
            "rootlens/data/processed/dataset_v1/"
            "processed_dataset_v1_metadata.json"
        ),
    )
    p.add_argument(
        "--split-metadata",
        type=Path,
        default=Path(
            "rootlens/data/processed/dataset_v1/splits/split_metadata.json"
        ),
    )
    p.add_argument(
        "--graph-config",
        type=Path,
        default=Path("rootlens/config/service_graph_v1.yaml"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rootlens/data/reports/graphsage_rca_v2"),
    )

    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-channels", type=int, default=48)
    p.add_argument("--mlp-hidden", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--learning-rate", type=float, default=7e-4)
    p.add_argument("--weight-decay", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    p.add_argument(
        "--skip-legacy-comparison",
        action="store_true",
        help="Do not mix this run with saved metrics from another split.",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    graph_dir = resolve_path(args.graph_dir, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root)
    dataset_metadata_path = resolve_path(args.dataset_metadata, repo_root)
    split_metadata_path = resolve_path(args.split_metadata, repo_root)
    graph_config_path = resolve_path(args.graph_config, repo_root)

    try:
        set_seed(args.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_path = graph_dir / "train_graphs.pt"
        val_path = graph_dir / "validation_graphs.pt"
        test_path = graph_dir / "test_graphs.pt"

        train_graphs = load_graphs(train_path)
        val_graphs = load_graphs(val_path)

        dataset_metadata = load_json(dataset_metadata_path)
        split_metadata = load_json(split_metadata_path)

        if not test_path.is_file():
            raise FileNotFoundError("Sealed test artifact is missing")

        validate_graphs(
            train_graphs,
            "train",
            int(split_metadata["row_counts_by_split"]["train"]),
            int(split_metadata["run_counts_by_split"]["train"]),
        )
        validate_graphs(
            val_graphs,
            "validation",
            int(split_metadata["row_counts_by_split"]["validation"]),
            int(split_metadata["run_counts_by_split"]["validation"]),
        )
        check_run_leakage(train_graphs, val_graphs)

        mean, std = compute_normalization(train_graphs)
        train_graphs = normalize_graphs(train_graphs, mean, std)
        val_graphs = normalize_graphs(val_graphs, mean, std)

        output_dir.mkdir(parents=True, exist_ok=True)

        normalization_path = output_dir / "feature_normalization.json"
        with normalization_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    metric: {
                        "mean": float(mean[i].item()),
                        "std": float(std[i].item()),
                    }
                    for i, metric in enumerate(NODE_FEATURES)
                },
                f,
                indent=2,
            )

        generator = torch.Generator().manual_seed(args.seed)

        train_loader = DataLoader(
            train_graphs,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_graphs,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        model = RootLensGraphSAGEV2(
            in_channels=NODE_FEATURE_COUNT,
            hidden_channels=args.hidden_channels,
            num_classes=len(CLASS_ORDER),
            dropout=args.dropout,
            mlp_hidden=args.mlp_hidden,
        ).to(device)

        weights = class_weights(train_graphs, device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        architecture_path = output_dir / "model_architecture.txt"
        architecture_path.write_text(str(model) + "\n", encoding="utf-8")

        parameter_count = sum(p.numel() for p in model.parameters())

        tracking_uri, db_path = configure_mlflow(
            repo_root,
            experiment_name=args.experiment_name,
        )

        best_f1 = -1.0
        best_epoch = 0
        best_state = None
        no_improvement = 0
        history_rows = []

        with mlflow.start_run(run_name="graphsage_rca_v2_node_preserving") as run:
            mlflow.set_tags(
                {
                    "project": "RootLensAI",
                    "task": "root_cause_service_classification",
                    "model_family": "GraphSAGE",
                    "architecture_variant": "node_identity_preserving",
                    "dataset_version": str(
                        dataset_metadata.get("dataset_version", "v1")
                    ),
                    "test_set_evaluated": "false",
                    "canonical_split": "true",
                    "evaluation_scope": "validation_only",
                    "device": str(device),
                }
            )

            mlflow.log_params(
                {
                    "model_name": "RootLensGraphSAGEV2",
                    "readout": "ordered_node_embedding_concatenation",
                    "seed": args.seed,
                    "epochs_max": args.epochs,
                    "patience": args.patience,
                    "batch_size": args.batch_size,
                    "hidden_channels": args.hidden_channels,
                    "mlp_hidden": args.mlp_hidden,
                    "dropout": args.dropout,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "num_sage_layers": 2,
                    "batch_norm": True,
                    "residual": True,
                    "optimizer": "AdamW",
                    "loss": "weighted_cross_entropy",
                    "parameter_count": parameter_count,
                    "node_count": NODE_COUNT,
                    "node_features": NODE_FEATURE_COUNT,
                    "train_graphs": len(train_graphs),
                    "validation_graphs": len(val_graphs),
                    "normalization": "train_only_standardization",
                    "git_commit": str(
                        dataset_metadata.get("source_git_commit", "unknown")
                    ),
                    "split_seed": int(split_metadata.get("split_seed", 42)),
                    "service_graph_sha256": sha256_file(graph_config_path),
                    "train_graphs_sha256": sha256_file(train_path),
                    "validation_graphs_sha256": sha256_file(val_path),
                    "torch_version": torch.__version__,
                    "cuda_runtime": str(torch.version.cuda),
                    "device_name": (
                        torch.cuda.get_device_name(0)
                        if device.type == "cuda"
                        else "CPU"
                    ),
                }
            )

            for epoch in range(1, args.epochs + 1):
                train_loss = train_epoch(
                    model, train_loader, optimizer, criterion, device
                )
                val = evaluate(model, val_loader, criterion, device)

                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val["loss"],
                    "val_accuracy": val["accuracy"],
                    "val_macro_f1": val["macro_f1"],
                    "val_weighted_f1": val["weighted_f1"],
                }
                history_rows.append(row)

                mlflow.log_metrics(
                    {
                        "train_loss": float(train_loss),
                        "val_loss": float(val["loss"]),
                        "val_accuracy": float(val["accuracy"]),
                        "val_macro_f1": float(val["macro_f1"]),
                        "val_weighted_f1": float(val["weighted_f1"]),
                    },
                    step=epoch,
                )

                print(
                    f"Epoch {epoch:03d} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val['loss']:.4f} | "
                    f"val_acc={val['accuracy']:.4f} | "
                    f"val_macro_f1={val['macro_f1']:.4f}"
                )

                if val["macro_f1"] > best_f1 + 1e-6:
                    best_f1 = float(val["macro_f1"])
                    best_epoch = epoch
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    no_improvement = 0
                else:
                    no_improvement += 1

                if no_improvement >= args.patience:
                    print(
                        f"Early stopping at epoch {epoch}; "
                        f"best epoch={best_epoch}, "
                        f"best macro F1={best_f1:.4f}"
                    )
                    break

            if best_state is None:
                raise RuntimeError("No best checkpoint was created")

            model.load_state_dict(best_state)
            model.to(device)
            best_val = evaluate(model, val_loader, criterion, device)

            mlflow.log_metrics(
                {
                    "best_epoch": float(best_epoch),
                    "best_val_accuracy": float(best_val["accuracy"]),
                    "best_val_macro_f1": float(best_val["macro_f1"]),
                    "best_val_weighted_f1": float(best_val["weighted_f1"]),
                    "best_val_loss": float(best_val["loss"]),
                }
            )

            report = classification_report(
                best_val["y_true"],
                best_val["y_pred"],
                labels=CLASS_ORDER,
                output_dict=True,
                zero_division=0,
            )

            for label in CLASS_ORDER:
                mlflow.log_metric(
                    f"val_precision_{label}",
                    float(report[label]["precision"]),
                )
                mlflow.log_metric(
                    f"val_recall_{label}",
                    float(report[label]["recall"]),
                )
                mlflow.log_metric(
                    f"val_f1_{label}",
                    float(report[label]["f1-score"]),
                )

            history = pd.DataFrame(history_rows)
            history_path = output_dir / "training_history.csv"
            history.to_csv(history_path, index=False)

            loss_path = output_dir / "training_loss_curve.png"
            metric_path = output_dir / "validation_metrics_curve.png"
            confusion_path = output_dir / "validation_confusion_matrix.png"

            save_loss_plot(history, loss_path)
            save_metric_plot(history, metric_path)
            save_confusion(
                best_val["y_true"],
                best_val["y_pred"],
                confusion_path,
            )

            report_json = output_dir / "validation_classification_report.json"
            report_csv = output_dir / "validation_classification_report.csv"

            with report_json.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            pd.DataFrame(report).transpose().to_csv(report_csv)

            predictions_path = output_dir / "validation_predictions.csv"
            pd.DataFrame(
                {
                    "run_id": best_val["run_ids"],
                    "timestamp": best_val["timestamps"],
                    "true_label": best_val["y_true"],
                    "predicted_label": best_val["y_pred"],
                    "confidence": best_val["confidence"],
                }
            ).to_csv(predictions_path, index=False)

            state_path = output_dir / "best_graphsage_rca_v2_state_dict.pt"
            torch.save(
                {
                    "state_dict": best_state,
                    "model_config": {
                        "in_channels": NODE_FEATURE_COUNT,
                        "hidden_channels": args.hidden_channels,
                        "num_classes": len(CLASS_ORDER),
                        "dropout": args.dropout,
                        "mlp_hidden": args.mlp_hidden,
                    },
                    "class_to_index": CLASS_TO_INDEX,
                    "feature_mean": mean,
                    "feature_std": std,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_f1,
                },
                state_path,
            )

            comparison_plot = output_dir / "validation_model_comparison.png"
            if args.skip_legacy_comparison:
                comparison = pd.DataFrame([{
                    "model": "graphsage_rca_v2_node_preserving",
                    "val_accuracy": float(best_val["accuracy"]),
                    "val_macro_f1": float(best_val["macro_f1"]),
                    "val_weighted_f1": float(best_val["weighted_f1"]),
                }])
                comparison.to_csv(comparison_plot.with_suffix(".csv"), index=False)
            else:
                comparison = save_model_comparison(
                    repo_root=repo_root,
                    v2_metrics={
                        "accuracy": float(best_val["accuracy"]),
                        "macro_f1": float(best_val["macro_f1"]),
                        "weighted_f1": float(best_val["weighted_f1"]),
                    },
                    path=comparison_plot,
                )

            summary = {
                "mlflow_run_id": run.info.run_id,
                "device": str(device),
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if device.type == "cuda"
                    else None
                ),
                "parameter_count": parameter_count,
                "best_epoch": best_epoch,
                "epochs_completed": len(history_rows),
                "best_validation_accuracy": float(best_val["accuracy"]),
                "best_validation_macro_f1": float(best_val["macro_f1"]),
                "best_validation_weighted_f1": float(
                    best_val["weighted_f1"]
                ),
                "test_set_evaluated": False,
                "comparison": comparison.to_dict(orient="records"),
            }

            summary_path = output_dir / "run_summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            for artifact in [
                normalization_path,
                architecture_path,
                history_path,
                loss_path,
                metric_path,
                confusion_path,
                report_json,
                report_csv,
                predictions_path,
                state_path,
                comparison_plot.with_suffix(".csv"),
                summary_path,
            ]:
                mlflow.log_artifact(
                    str(artifact),
                    artifact_path="graphsage_rca_v2",
                )

            if not args.skip_legacy_comparison:
                mlflow.log_artifact(
                    str(comparison_plot),
                    artifact_path="graphsage_rca_v2",
                )

            print()
            print("=" * 94)
            print("RootLens GraphSAGE RCA v2 — Training Complete")
            print("=" * 94)
            print(f"Device:                    {device}")
            print(
                f"GPU:                       "
                f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'}"
            )
            print(f"Model parameters:          {parameter_count:,}")
            print(f"Best epoch:                {best_epoch}")
            print(
                f"Validation accuracy:       {best_val['accuracy']:.4f}"
            )
            print(
                f"Validation macro F1:       {best_val['macro_f1']:.4f}"
            )
            print(
                f"Validation weighted F1:    {best_val['weighted_f1']:.4f}"
            )
            print()
            print("MODEL COMPARISON")
            print("-" * 94)
            for _, r in comparison.iterrows():
                print(
                    f"{r['model']:<38} "
                    f"macro_f1={r['val_macro_f1']:.4f}"
                )
            print()
            print(f"Comparison chart:          {comparison_plot}")
            print(f"Confusion matrix:          {confusion_path}")
            print(f"Training curves:           {loss_path}")
            print(f"                           {metric_path}")
            print()
            print(f"MLflow run ID:             {run.info.run_id}")
            print(f"MLflow tracking URI:       {tracking_uri}")
            print()
            print("TEST SET REMAINS SEALED.")
            print("=" * 94)
            print()

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
