#!/usr/bin/env python3
"""
Train the first RootLens GraphSAGE RCA model with CUDA + MLflow.

Research policy:
- Train on train_graphs.pt only.
- Use validation_graphs.pt for early stopping / model selection.
- DO NOT evaluate test_graphs.pt here.
- Fit feature normalization statistics on TRAIN nodes only.
- Log all important parameters, metrics, plots, and model artifacts to MLflow.

Model:
    12 nodes × 7 telemetry features
        -> GraphSAGE(7 -> 32)
        -> ReLU
        -> Dropout
        -> GraphSAGE(32 -> 32)
        -> ReLU
        -> Global Mean Pool
        -> Linear(32 -> 5 classes)

Outputs:
    rootlens/data/reports/graphsage_v1/
        training_history.csv
        training_loss_curve.png
        validation_metrics_curve.png
        validation_confusion_matrix.png
        validation_classification_report.csv
        validation_classification_report.json
        validation_predictions.csv
        feature_normalization.json
        model_architecture.txt
        best_graphsage_state_dict.pt
        run_summary.json

MLflow:
    Experiment: RootLens-RCA-GNN-v1
    Backend:   rootlens/mlflow.db

Usage:
    python rootlens\\scripts\\train_graphsage_mlflow.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
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
from torch_geometric.nn import SAGEConv, global_mean_pool


EXPERIMENT_NAME = "RootLens-RCA-GNN-v1"

CLASS_TO_INDEX = {
    "healthy": 0,
    "payment": 1,
    "cart": 2,
    "checkout": 3,
    "product_catalog": 4,
}

INDEX_TO_CLASS = {v: k for k, v in CLASS_TO_INDEX.items()}

CLASS_ORDER = [
    "healthy",
    "payment",
    "cart",
    "checkout",
    "product_catalog",
]

NODE_FEATURES = [
    "cpu",
    "memory",
    "request_rate",
    "has_requests",
    "latency_ms",
    "error_rps",
    "error_rate",
]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility where practical.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path: Path, repo_root: Path) -> Path:
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def load_graphs(path: Path) -> list[Data]:
    if not path.is_file():
        raise FileNotFoundError(f"Graph artifact not found: {path}")

    graphs = torch.load(path, weights_only=False)

    if not isinstance(graphs, list):
        raise ValueError(
            f"Expected a list of PyG Data objects in {path}, "
            f"got {type(graphs)}"
        )

    if not graphs:
        raise ValueError(f"No graphs found in {path}")

    return graphs


# ---------------------------------------------------------------------------
# Graph validation
# ---------------------------------------------------------------------------

def validate_graphs(
    graphs: list[Data],
    split_name: str,
    expected_graphs: int,
    expected_runs: int,
) -> None:
    if len(graphs) != expected_graphs:
        raise ValueError(
            f"{split_name}: expected {expected_graphs} graphs, "
            f"found {len(graphs)}"
        )

    run_ids = set()
    observed_labels = set()

    reference_edge_index = graphs[0].edge_index

    for idx, graph in enumerate(graphs):
        if tuple(graph.x.shape) != (12, 7):
            raise ValueError(
                f"{split_name} graph {idx}: expected x shape (12, 7), "
                f"found {tuple(graph.x.shape)}"
            )

        if torch.isnan(graph.x).any():
            raise ValueError(
                f"{split_name} graph {idx}: node features contain NaN"
            )

        if not torch.equal(graph.edge_index, reference_edge_index):
            raise ValueError(
                f"{split_name} graph {idx}: edge_index differs "
                "from frozen topology"
            )

        label_idx = int(graph.y.item())
        if label_idx not in INDEX_TO_CLASS:
            raise ValueError(
                f"{split_name} graph {idx}: unknown label {label_idx}"
            )

        observed_labels.add(INDEX_TO_CLASS[label_idx])
        run_ids.add(str(graph.run_id))

    if len(run_ids) != expected_runs:
        raise ValueError(
            f"{split_name}: expected {expected_runs} runs, found {len(run_ids)}"
        )

    if observed_labels != set(CLASS_ORDER):
        raise ValueError(
            f"{split_name}: class coverage mismatch. "
            f"Observed={sorted(observed_labels)}"
        )


def validate_no_run_leakage(
    train_graphs: list[Data],
    val_graphs: list[Data],
) -> None:
    train_runs = {str(g.run_id) for g in train_graphs}
    val_runs = {str(g.run_id) for g in val_graphs}

    overlap = train_runs & val_runs

    if overlap:
        raise ValueError(
            f"RUN LEAKAGE between train and validation: {sorted(overlap)}"
        )


# ---------------------------------------------------------------------------
# Feature normalization
# ---------------------------------------------------------------------------

def compute_train_normalization(
    graphs: list[Data],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fit mean/std ONLY from training-node telemetry.

    Shape after concatenation:
        [num_train_graphs * 12, 7]
    """
    all_x = torch.cat(
        [graph.x for graph in graphs],
        dim=0,
    ).float()

    mean = all_x.mean(dim=0)
    std = all_x.std(dim=0, unbiased=False)

    # Avoid division by zero for constant features.
    std = torch.where(
        std < 1e-8,
        torch.ones_like(std),
        std,
    )

    return mean, std


def normalize_graphs(
    graphs: list[Data],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> list[Data]:
    """
    Return deep-copied graphs so frozen serialized graph artifacts stay unchanged.
    """
    output: list[Data] = []

    for graph in graphs:
        g = copy.deepcopy(graph)
        g.x = (g.x.float() - mean) / std

        if torch.isnan(g.x).any() or torch.isinf(g.x).any():
            raise ValueError(
                f"Normalization produced NaN/Inf for run {g.run_id}"
            )

        output.append(g)

    return output


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RootLensGraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        hidden_channels: int = 32,
        num_classes: int = 5,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        self.conv1 = SAGEConv(
            in_channels,
            hidden_channels,
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            hidden_channels,
        )

        self.dropout = dropout

        self.classifier = nn.Linear(
            hidden_channels,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Convert node embeddings -> one graph embedding per telemetry window.
        x = global_mean_pool(x, batch)

        logits = self.classifier(x)
        return logits


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def build_class_weights(
    graphs: list[Data],
    device: torch.device,
) -> torch.Tensor:
    counts = np.zeros(len(CLASS_ORDER), dtype=np.int64)

    for graph in graphs:
        counts[int(graph.y.item())] += 1

    if np.any(counts == 0):
        raise ValueError(
            f"Cannot build class weights; zero-count class: {counts.tolist()}"
        )

    total = counts.sum()
    n_classes = len(counts)

    # sklearn-style balanced weighting:
    # total / (n_classes * class_count)
    weights = total / (n_classes * counts.astype(np.float64))

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()

    running_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            batch.x,
            batch.edge_index,
            batch.batch,
        )

        targets = batch.y.view(-1)

        loss = criterion(
            logits,
            targets,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        batch_size = int(targets.shape[0])
        running_loss += float(loss.item()) * batch_size
        total_graphs += batch_size

    return running_loss / max(total_graphs, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()

    running_loss = 0.0
    total_graphs = 0

    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []

    run_ids: list[str] = []
    timestamps: list[str] = []

    for batch in loader:
        batch = batch.to(device)

        logits = model(
            batch.x,
            batch.edge_index,
            batch.batch,
        )

        targets = batch.y.view(-1)

        loss = criterion(
            logits,
            targets,
        )

        probs = torch.softmax(
            logits,
            dim=1,
        )

        predictions = probs.argmax(dim=1)
        confidence = probs.max(dim=1).values

        batch_size = int(targets.shape[0])
        running_loss += float(loss.item()) * batch_size
        total_graphs += batch_size

        y_true.extend(
            targets.detach().cpu().tolist()
        )
        y_pred.extend(
            predictions.detach().cpu().tolist()
        )
        confidences.extend(
            confidence.detach().cpu().tolist()
        )

        # PyG batches arbitrary string attrs into Python lists.
        run_ids.extend(
            [str(v) for v in batch.run_id]
        )
        timestamps.extend(
            [str(v) for v in batch.timestamp]
        )

    true_names = [
        INDEX_TO_CLASS[idx]
        for idx in y_true
    ]
    pred_names = [
        INDEX_TO_CLASS[idx]
        for idx in y_pred
    ]

    return {
        "loss": running_loss / max(total_graphs, 1),
        "accuracy": accuracy_score(
            true_names,
            pred_names,
        ),
        "macro_f1": f1_score(
            true_names,
            pred_names,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            true_names,
            pred_names,
            labels=CLASS_ORDER,
            average="weighted",
            zero_division=0,
        ),
        "y_true_idx": y_true,
        "y_pred_idx": y_pred,
        "y_true": true_names,
        "y_pred": pred_names,
        "confidence": confidences,
        "run_ids": run_ids,
        "timestamps": timestamps,
    }


# ---------------------------------------------------------------------------
# Plot / report helpers
# ---------------------------------------------------------------------------

def save_training_loss_plot(
    history: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        history["epoch"],
        history["train_loss"],
        label="Train loss",
    )
    ax.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation loss",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("RootLens GraphSAGE Training Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_validation_metrics_plot(
    history: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        history["epoch"],
        history["val_accuracy"],
        label="Validation accuracy",
    )
    ax.plot(
        history["epoch"],
        history["val_macro_f1"],
        label="Validation macro F1",
    )
    ax.plot(
        history["epoch"],
        history["val_weighted_f1"],
        label="Validation weighted F1",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("RootLens GraphSAGE Validation Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    path: Path,
) -> None:
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_ORDER,
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    display.plot(
        ax=ax,
        xticks_rotation=35,
        values_format="d",
        colorbar=False,
    )
    ax.set_title(
        "RootLens GraphSAGE Validation Confusion Matrix"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_classification_report(
    y_true: list[str],
    y_pred: list[str],
    json_path: Path,
    csv_path: Path,
) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    pd.DataFrame(
        report
    ).transpose().to_csv(
        csv_path
    )

    return report


def save_predictions(
    evaluation: dict[str, Any],
    path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "run_id": evaluation["run_ids"],
            "timestamp": evaluation["timestamps"],
            "true_label": evaluation["y_true"],
            "predicted_label": evaluation["y_pred"],
            "confidence": evaluation["confidence"],
        }
    )

    df.to_csv(
        path,
        index=False,
    )


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

def configure_mlflow(
    repo_root: Path,
    experiment_name: str = EXPERIMENT_NAME,
) -> tuple[str, Path]:
    db_path = (
        repo_root
        / "rootlens"
        / "mlflow.db"
    ).resolve()

    tracking_uri = (
        f"sqlite:///{db_path.as_posix()}"
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        experiment_name
    )

    return tracking_uri, db_path


def log_per_class_metrics(
    report: dict[str, Any],
) -> None:
    for label in CLASS_ORDER:
        values = report[label]

        mlflow.log_metric(
            f"val_precision_{label}",
            float(values["precision"]),
        )
        mlflow.log_metric(
            f"val_recall_{label}",
            float(values["recall"]),
        )
        mlflow.log_metric(
            f"val_f1_{label}",
            float(values["f1-score"]),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RootLens GraphSAGE v1 with MLflow."
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
    )

    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path(
            "rootlens/data/processed/dataset_v1/graphs"
        ),
    )

    parser.add_argument(
        "--dataset-metadata",
        type=Path,
        default=Path(
            "rootlens/data/processed/dataset_v1/"
            "processed_dataset_v1_metadata.json"
        ),
    )

    parser.add_argument(
        "--split-metadata",
        type=Path,
        default=Path(
            "rootlens/data/processed/dataset_v1/splits/"
            "split_metadata.json"
        ),
    )

    parser.add_argument(
        "--graph-config",
        type=Path,
        default=Path(
            "rootlens/config/service_graph_v1.yaml"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "rootlens/data/reports/graphsage_v1"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--experiment-name",
        default=EXPERIMENT_NAME,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("rootlens/data/manifests/dataset_v1.yaml"),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    graph_dir = resolve_path(args.graph_dir, repo_root)
    dataset_metadata_path = resolve_path(
        args.dataset_metadata,
        repo_root,
    )
    split_metadata_path = resolve_path(
        args.split_metadata,
        repo_root,
    )
    graph_config_path = resolve_path(
        args.graph_config,
        repo_root,
    )
    manifest_path = resolve_path(args.manifest, repo_root)
    output_dir = resolve_path(
        args.output_dir,
        repo_root,
    )

    try:
        set_seed(args.seed)

        # --------------------------------------------------------------
        # Device
        # --------------------------------------------------------------

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # --------------------------------------------------------------
        # Inputs
        # --------------------------------------------------------------

        train_path = graph_dir / "train_graphs.pt"
        val_path = graph_dir / "validation_graphs.pt"
        test_path = graph_dir / "test_graphs.pt"
        graph_metadata_path = (
            graph_dir / "graph_dataset_metadata.json"
        )

        dataset_metadata = load_json(dataset_metadata_path)
        split_metadata = load_json(split_metadata_path)
        graph_metadata = load_json(graph_metadata_path)

        train_graphs = load_graphs(train_path)
        val_graphs = load_graphs(val_path)

        # Test artifact must exist, but remains sealed.
        if not test_path.is_file():
            raise FileNotFoundError(
                f"Sealed test graph artifact missing: {test_path}"
            )

        validate_graphs(
            train_graphs,
            split_name="train",
            expected_graphs=int(split_metadata["row_counts_by_split"]["train"]),
            expected_runs=int(split_metadata["run_counts_by_split"]["train"]),
        )

        validate_graphs(
            val_graphs,
            split_name="validation",
            expected_graphs=int(split_metadata["row_counts_by_split"]["validation"]),
            expected_runs=int(split_metadata["run_counts_by_split"]["validation"]),
        )

        validate_no_run_leakage(
            train_graphs,
            val_graphs,
        )

        # --------------------------------------------------------------
        # Train-only normalization
        # --------------------------------------------------------------

        feature_mean, feature_std = (
            compute_train_normalization(
                train_graphs
            )
        )

        train_graphs_norm = normalize_graphs(
            train_graphs,
            mean=feature_mean,
            std=feature_std,
        )

        val_graphs_norm = normalize_graphs(
            val_graphs,
            mean=feature_mean,
            std=feature_std,
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        normalization_path = (
            output_dir
            / "feature_normalization.json"
        )

        normalization_payload = {
            feature: {
                "mean": float(feature_mean[idx].item()),
                "std": float(feature_std[idx].item()),
            }
            for idx, feature
            in enumerate(NODE_FEATURES)
        }

        with normalization_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                normalization_payload,
                f,
                indent=2,
            )

        # --------------------------------------------------------------
        # Loaders
        # --------------------------------------------------------------

        train_generator = torch.Generator()
        train_generator.manual_seed(
            args.seed
        )

        train_loader = DataLoader(
            train_graphs_norm,
            batch_size=args.batch_size,
            shuffle=True,
            generator=train_generator,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_graphs_norm,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # --------------------------------------------------------------
        # Model / objective
        # --------------------------------------------------------------

        model = RootLensGraphSAGE(
            in_channels=7,
            hidden_channels=args.hidden_channels,
            num_classes=len(CLASS_ORDER),
            dropout=args.dropout,
        ).to(device)

        class_weights = build_class_weights(
            train_graphs_norm,
            device=device,
        )

        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        architecture_path = (
            output_dir
            / "model_architecture.txt"
        )

        architecture_path.write_text(
            str(model) + "\n",
            encoding="utf-8",
        )

        # --------------------------------------------------------------
        # MLflow
        # --------------------------------------------------------------

        tracking_uri, db_path = configure_mlflow(
            repo_root,
            experiment_name=args.experiment_name,
        )

        history_rows: list[dict[str, Any]] = []

        best_macro_f1 = -1.0
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0

        with mlflow.start_run(
            run_name="graphsage_v1"
        ) as run:

            mlflow.set_tags(
                {
                    "project": "RootLensAI",
                    "task": "root_cause_service_classification",
                    "model_family": "GraphSAGE",
                    "dataset_version": str(
                        dataset_metadata.get("dataset_version", "v1")
                    ),
                    "graph_version": str(
                        graph_metadata.get(
                            "source_graph_version",
                            "v1",
                        )
                    ),
                    "sample_unit": "system_wide_telemetry_graph",
                    "test_set_evaluated": "false",
                    "device": str(device),
                }
            )

            mlflow.log_params(
                {
                    "model_name": "RootLensGraphSAGE",
                    "seed": args.seed,
                    "epochs_max": args.epochs,
                    "early_stopping_patience": args.patience,
                    "batch_size": args.batch_size,
                    "hidden_channels": args.hidden_channels,
                    "num_sage_layers": 2,
                    "dropout": args.dropout,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "optimizer": "AdamW",
                    "loss": "weighted_cross_entropy",
                    "pooling": "global_mean_pool",
                    "node_count": 12,
                    "node_feature_count": 7,
                    "num_classes": 5,
                    "train_graphs": len(train_graphs_norm),
                    "validation_graphs": len(val_graphs_norm),
                    "train_runs": len({g.run_id for g in train_graphs_norm}),
                    "validation_runs": len({g.run_id for g in val_graphs_norm}),
                    "dataset_manifest_path": str(manifest_path),
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "normalization": "train_only_standardization",
                    "git_commit": str(
                        dataset_metadata.get(
                            "source_git_commit",
                            "unknown",
                        )
                    ),
                    "split_seed": int(
                        split_metadata.get(
                            "split_seed",
                            42,
                        )
                    ),
                    "service_graph_sha256": sha256_file(
                        graph_config_path
                    ),
                    "train_graphs_sha256": sha256_file(
                        train_path
                    ),
                    "validation_graphs_sha256": sha256_file(
                        val_path
                    ),
                    "torch_version": torch.__version__,
                    "cuda_runtime": str(torch.version.cuda),
                    "device_name": (
                        torch.cuda.get_device_name(0)
                        if device.type == "cuda"
                        else "CPU"
                    ),
                }
            )

            # ----------------------------------------------------------
            # Training loop
            # ----------------------------------------------------------

            training_started = time.perf_counter()
            for epoch in range(
                1,
                args.epochs + 1,
            ):
                train_loss = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                )

                val_metrics = evaluate(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                )

                history_rows.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
                        "val_macro_f1": val_metrics["macro_f1"],
                        "val_weighted_f1": val_metrics["weighted_f1"],
                    }
                )

                mlflow.log_metrics(
                    {
                        "train_loss": float(train_loss),
                        "val_loss": float(val_metrics["loss"]),
                        "val_accuracy": float(val_metrics["accuracy"]),
                        "val_macro_f1": float(val_metrics["macro_f1"]),
                        "val_weighted_f1": float(
                            val_metrics["weighted_f1"]
                        ),
                    },
                    step=epoch,
                )

                print(
                    f"Epoch {epoch:03d} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"val_acc={val_metrics['accuracy']:.4f} | "
                    f"val_macro_f1={val_metrics['macro_f1']:.4f}"
                )

                if val_metrics["macro_f1"] > best_macro_f1 + 1e-6:
                    best_macro_f1 = float(
                        val_metrics["macro_f1"]
                    )
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value
                        in model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= args.patience:
                    print(
                        f"Early stopping at epoch {epoch}; "
                        f"best epoch={best_epoch}, "
                        f"best val macro F1={best_macro_f1:.4f}"
                    )
                    break

            if best_state is None:
                raise RuntimeError(
                    "Training finished without a best model checkpoint"
                )

            # ----------------------------------------------------------
            # Restore best model
            # ----------------------------------------------------------

            model.load_state_dict(best_state)
            model.to(device)

            best_val = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
            training_duration_seconds = time.perf_counter() - training_started

            mlflow.log_metrics(
                {
                    "best_epoch": float(best_epoch),
                    "best_val_accuracy": float(
                        best_val["accuracy"]
                    ),
                    "best_val_macro_f1": float(
                        best_val["macro_f1"]
                    ),
                    "best_val_weighted_f1": float(
                        best_val["weighted_f1"]
                    ),
                    "best_val_loss": float(
                        best_val["loss"]
                    ),
                    "training_duration_seconds": float(training_duration_seconds),
                }
            )

            # ----------------------------------------------------------
            # Persist artifacts
            # ----------------------------------------------------------

            history_df = pd.DataFrame(
                history_rows
            )

            history_path = (
                output_dir
                / "training_history.csv"
            )
            history_df.to_csv(
                history_path,
                index=False,
            )

            loss_plot_path = (
                output_dir
                / "training_loss_curve.png"
            )
            save_training_loss_plot(
                history_df,
                loss_plot_path,
            )

            metrics_plot_path = (
                output_dir
                / "validation_metrics_curve.png"
            )
            save_validation_metrics_plot(
                history_df,
                metrics_plot_path,
            )

            confusion_path = (
                output_dir
                / "validation_confusion_matrix.png"
            )
            save_confusion_matrix(
                best_val["y_true"],
                best_val["y_pred"],
                confusion_path,
            )

            report_json_path = (
                output_dir
                / "validation_classification_report.json"
            )
            report_csv_path = (
                output_dir
                / "validation_classification_report.csv"
            )

            report = save_classification_report(
                best_val["y_true"],
                best_val["y_pred"],
                report_json_path,
                report_csv_path,
            )

            log_per_class_metrics(
                report
            )

            predictions_path = (
                output_dir
                / "validation_predictions.csv"
            )
            save_predictions(
                best_val,
                predictions_path,
            )

            state_dict_path = (
                output_dir
                / "best_graphsage_state_dict.pt"
            )

            torch.save(
                {
                    "state_dict": best_state,
                    "model_config": {
                        "in_channels": 7,
                        "hidden_channels": args.hidden_channels,
                        "num_classes": 5,
                        "dropout": args.dropout,
                    },
                    "class_to_index": CLASS_TO_INDEX,
                    "node_features": NODE_FEATURES,
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_macro_f1,
                },
                state_dict_path,
            )

            summary = {
                "mlflow_run_id": run.info.run_id,
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(0)
                    if device.type == "cuda"
                    else "CPU"
                ),
                "best_epoch": best_epoch,
                "epochs_completed": len(history_rows),
                "best_validation_accuracy": float(
                    best_val["accuracy"]
                ),
                "best_validation_macro_f1": float(
                    best_val["macro_f1"]
                ),
                "best_validation_weighted_f1": float(
                    best_val["weighted_f1"]
                ),
                "best_validation_loss": float(
                    best_val["loss"]
                ),
                "test_set_evaluated": False,
                "random_forest_validation_macro_f1_reference": 0.7600,
            }

            summary_path = (
                output_dir
                / "run_summary.json"
            )

            with summary_path.open(
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    summary,
                    f,
                    indent=2,
                )

            # Log all locally generated artifacts.
            for artifact in [
                history_path,
                loss_plot_path,
                metrics_plot_path,
                confusion_path,
                report_json_path,
                report_csv_path,
                predictions_path,
                normalization_path,
                architecture_path,
                state_dict_path,
                summary_path,
            ]:
                mlflow.log_artifact(
                    str(artifact),
                    artifact_path="graphsage_v1",
                )

            # ----------------------------------------------------------
            # Final console summary
            # ----------------------------------------------------------

            print()
            print("=" * 92)
            print("RootLens GraphSAGE v1 — MLflow Training Complete")
            print("=" * 92)
            print(f"Device:                    {device}")
            print(
                f"GPU:                       "
                f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'}"
            )
            print(f"Best epoch:                {best_epoch}")
            print(
                f"Validation accuracy:       "
                f"{best_val['accuracy']:.4f}"
            )
            print(
                f"Validation macro F1:       "
                f"{best_val['macro_f1']:.4f}"
            )
            print(
                f"Validation weighted F1:    "
                f"{best_val['weighted_f1']:.4f}"
            )
            print()
            print(
                f"Random Forest reference:   "
                f"macro F1 = 0.7600"
            )
            print()
            print(f"MLflow run ID:             {run.info.run_id}")
            print(f"MLflow tracking URI:       {tracking_uri}")
            print(f"MLflow database:           {db_path}")
            print()
            print("VISUALIZATIONS")
            print("-" * 92)
            print(f"Loss curve:        {loss_plot_path}")
            print(f"Metric curves:     {metrics_plot_path}")
            print(f"Confusion matrix:  {confusion_path}")
            print()
            print("TEST SET REMAINS SEALED.")
            print("=" * 92)
            print()

        return 0

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
