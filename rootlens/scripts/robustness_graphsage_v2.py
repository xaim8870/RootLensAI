#scripts/robustness_graphsage_v2.py
#!/usr/bin/env python3
"""
RootLens GraphSAGE RCA v2 robustness + topology ablation study.

Purpose
-------
Before promoting the node-preserving GraphSAGE v2 to the MVP:

1. Train the same architecture across multiple random seeds.
2. Report window-level validation metrics.
3. Report RUN-level validation metrics using majority vote.
4. Compare REAL service topology vs SELF-LOOPS-ONLY ablation.
5. Keep test_graphs.pt completely sealed.
6. Log each experiment to MLflow.
7. Generate presentation-ready robustness visualizations.

Default seeds:
    11, 21, 42, 73, 101

Conditions:
    real_graph
    self_loops_only

Total default experiments:
    5 seeds × 2 topology conditions = 10 runs

This is deliberately NOT a hyperparameter sweep.
The architecture/hyperparameters remain fixed; only seed and topology condition vary.

Usage:
    python rootlens\\scripts\\robustness_graphsage_v2.py
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, SAGEConv


EXPERIMENT_NAME = "RootLens-RCA-GNN-v2-Robustness"

CLASS_TO_INDEX = {
    "healthy": 0,
    "payment": 1,
    "cart": 2,
    "checkout": 3,
    "product_catalog": 4,
}
INDEX_TO_CLASS = {v: k for k, v in CLASS_TO_INDEX.items()}
CLASS_ORDER = ["healthy", "payment", "cart", "checkout", "product_catalog"]

NODE_COUNT = 12
NODE_FEATURE_COUNT = 7

DEFAULT_SEEDS = [11, 21, 42, 73, 101]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(path: Path, root: Path) -> Path:
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_graphs(path: Path) -> list[Data]:
    graphs = torch.load(path, weights_only=False)

    if not isinstance(graphs, list) or not graphs:
        raise ValueError(f"Invalid graph artifact: {path}")

    return graphs


def validate_no_leakage(
    train_graphs: list[Data],
    val_graphs: list[Data],
) -> None:
    train_runs = {str(g.run_id) for g in train_graphs}
    val_runs = {str(g.run_id) for g in val_graphs}

    overlap = train_runs & val_runs
    if overlap:
        raise ValueError(f"Run leakage detected: {sorted(overlap)}")


def compute_normalization(
    graphs: list[Data],
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.cat([g.x.float() for g in graphs], dim=0)

    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)

    std = torch.where(
        std < 1e-8,
        torch.ones_like(std),
        std,
    )

    return mean, std


def normalized_copy(
    graphs: list[Data],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> list[Data]:
    output = []

    for graph in graphs:
        g = copy.deepcopy(graph)
        g.x = (g.x.float() - mean) / std
        output.append(g)

    return output


def make_self_loop_edge_index() -> torch.Tensor:
    idx = torch.arange(NODE_COUNT, dtype=torch.long)

    return torch.stack(
        [idx, idx],
        dim=0,
    )


def apply_topology_condition(
    graphs: list[Data],
    condition: str,
) -> list[Data]:
    """
    real_graph:
        retain frozen bidirectional graph + self-loops.

    self_loops_only:
        remove all inter-service edges while preserving each node's
        own GraphSAGE transformation.
    """
    if condition == "real_graph":
        return [copy.deepcopy(g) for g in graphs]

    if condition == "self_loops_only":
        edge_index = make_self_loop_edge_index()
        output = []

        for graph in graphs:
            g = copy.deepcopy(graph)
            g.edge_index = edge_index.clone()
            output.append(g)

        return output

    raise ValueError(f"Unknown topology condition: {condition}")


class RootLensGraphSAGEV2(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 48,
        dropout: float = 0.30,
        mlp_hidden: int = 96,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels
        self.dropout_p = dropout

        self.conv1 = SAGEConv(
            NODE_FEATURE_COUNT,
            hidden_channels,
        )
        self.bn1 = BatchNorm(hidden_channels)

        self.conv2 = SAGEConv(
            hidden_channels,
            hidden_channels,
        )
        self.bn2 = BatchNorm(hidden_channels)

        self.classifier = nn.Sequential(
            nn.Linear(
                NODE_COUNT * hidden_channels,
                mlp_hidden,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                mlp_hidden,
                len(CLASS_ORDER),
            ),
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
        x1 = F.dropout(
            x1,
            p=self.dropout_p,
            training=self.training,
        )

        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        x = x1 + x2

        batch_size = int(batch.max().item()) + 1

        if x.shape[0] != batch_size * NODE_COUNT:
            raise RuntimeError(
                "Node count/order invariant violated"
            )

        x = x.view(
            batch_size,
            NODE_COUNT * self.hidden_channels,
        )

        return self.classifier(x)


def build_class_weights(
    graphs: list[Data],
    device: torch.device,
) -> torch.Tensor:
    counts = np.zeros(
        len(CLASS_ORDER),
        dtype=np.int64,
    )

    for graph in graphs:
        counts[int(graph.y.item())] += 1

    total = counts.sum()

    weights = total / (
        len(CLASS_ORDER)
        * counts.astype(np.float64)
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


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

        n = int(targets.shape[0])
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

    true_idx = []
    pred_idx = []
    run_ids = []

    for batch in loader:
        batch = batch.to(device)

        logits = model(
            batch.x,
            batch.edge_index,
            batch.batch,
        )

        targets = batch.y.view(-1)
        loss = criterion(logits, targets)

        preds = logits.argmax(dim=1)

        n = int(targets.shape[0])
        total_loss += float(loss.item()) * n
        total += n

        true_idx.extend(
            targets.cpu().tolist()
        )
        pred_idx.extend(
            preds.cpu().tolist()
        )
        run_ids.extend(
            [str(v) for v in batch.run_id]
        )

    y_true = [
        INDEX_TO_CLASS[i]
        for i in true_idx
    ]

    y_pred = [
        INDEX_TO_CLASS[i]
        for i in pred_idx
    ]

    return {
        "loss": total_loss / max(total, 1),
        "window_accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "window_macro_f1": f1_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        ),
        "window_weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="weighted",
            zero_division=0,
        ),
        "y_true": y_true,
        "y_pred": y_pred,
        "run_ids": run_ids,
    }


def compute_run_level_metrics(
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """
    Majority-vote the 60 telemetry-window predictions belonging to each run.
    """

    true_by_run: dict[str, list[str]] = defaultdict(list)
    pred_by_run: dict[str, list[str]] = defaultdict(list)

    for run_id, true_label, pred_label in zip(
        evaluation["run_ids"],
        evaluation["y_true"],
        evaluation["y_pred"],
    ):
        true_by_run[run_id].append(true_label)
        pred_by_run[run_id].append(pred_label)

    run_rows = []

    for run_id in sorted(true_by_run):
        true_values = true_by_run[run_id]
        pred_values = pred_by_run[run_id]

        if len(set(true_values)) != 1:
            raise ValueError(
                f"Inconsistent true label in validation run {run_id}"
            )

        true_label = true_values[0]

        counts = Counter(pred_values)

        # deterministic tie-breaking with CLASS_ORDER
        predicted_label = max(
            CLASS_ORDER,
            key=lambda label: (
                counts.get(label, 0),
                -CLASS_ORDER.index(label),
            ),
        )

        correct_windows = sum(
            p == true_label
            for p in pred_values
        )

        run_rows.append(
            {
                "run_id": run_id,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct_windows": correct_windows,
                "total_windows": len(pred_values),
                "window_accuracy_within_run": (
                    correct_windows
                    / len(pred_values)
                ),
            }
        )

    run_df = pd.DataFrame(run_rows)

    run_accuracy = accuracy_score(
        run_df["true_label"],
        run_df["predicted_label"],
    )

    run_macro_f1 = f1_score(
        run_df["true_label"],
        run_df["predicted_label"],
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0,
    )

    return {
        "run_accuracy": float(run_accuracy),
        "run_macro_f1": float(run_macro_f1),
        "correct_runs": int(
            (
                run_df["true_label"]
                == run_df["predicted_label"]
            ).sum()
        ),
        "total_runs": len(run_df),
        "run_predictions": run_df,
    }


def configure_mlflow(
    repo_root: Path,
) -> str:
    db_path = (
        repo_root
        / "rootlens"
        / "mlflow.db"
    ).resolve()

    uri = f"sqlite:///{db_path.as_posix()}"

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    return uri


def run_experiment(
    *,
    seed: int,
    condition: str,
    base_train: list[Data],
    base_val: list[Data],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    hidden_channels: int,
    mlp_hidden: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    set_seed(seed)

    train_graphs = apply_topology_condition(
        normalized_copy(
            base_train,
            mean,
            std,
        ),
        condition,
    )

    val_graphs = apply_topology_condition(
        normalized_copy(
            base_val,
            mean,
            std,
        ),
        condition,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = RootLensGraphSAGEV2(
        hidden_channels=hidden_channels,
        dropout=dropout,
        mlp_hidden=mlp_hidden,
    ).to(device)

    weights = build_class_weights(
        train_graphs,
        device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    no_improvement = 0

    for epoch in range(
        1,
        epochs + 1,
    ):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        if val["window_macro_f1"] > best_f1 + 1e-6:
            best_f1 = float(
                val["window_macro_f1"]
            )
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            "No best checkpoint created"
        )

    model.load_state_dict(best_state)
    model.to(device)

    final_val = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )

    run_level = compute_run_level_metrics(
        final_val
    )

    return {
        "seed": seed,
        "condition": condition,
        "best_epoch": best_epoch,
        "window_accuracy": float(
            final_val["window_accuracy"]
        ),
        "window_macro_f1": float(
            final_val["window_macro_f1"]
        ),
        "window_weighted_f1": float(
            final_val["window_weighted_f1"]
        ),
        "run_accuracy": float(
            run_level["run_accuracy"]
        ),
        "run_macro_f1": float(
            run_level["run_macro_f1"]
        ),
        "correct_runs": int(
            run_level["correct_runs"]
        ),
        "total_runs": int(
            run_level["total_runs"]
        ),
        "run_predictions": (
            run_level["run_predictions"]
        ),
    }


def save_seed_plot(
    results: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for condition, group in results.groupby(
        "condition"
    ):
        group = group.sort_values("seed")

        ax.plot(
            group["seed"].astype(str),
            group["window_macro_f1"],
            marker="o",
            label=condition,
        )

    ax.axhline(
        0.9683,
        linestyle="--",
        label="Random Forest baseline",
    )

    ax.set_ylim(0.80, 1.01)
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Validation macro F1")
    ax.set_title(
        "RootLens GraphSAGE v2 Robustness Across Seeds"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_ablation_plot(
    summary: pd.DataFrame,
    path: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    labels = summary["condition"].tolist()
    means = summary[
        "window_macro_f1_mean"
    ].tolist()

    ax.bar(
        labels,
        means,
    )

    ax.axhline(
        0.9683,
        linestyle="--",
        label="Random Forest baseline",
    )

    ax.set_ylim(0.80, 1.01)
    ax.set_ylabel(
        "Mean validation macro F1"
    )
    ax.set_title(
        "RootLens Topology Ablation"
    )

    for i, value in enumerate(means):
        ax.text(
            i,
            min(value + 0.004, 1.005),
            f"{value:.4f}",
            ha="center",
        )

    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GraphSAGE v2 multi-seed robustness "
            "and topology ablation."
        )
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
            "rootlens/data/processed/"
            "dataset_v1/graphs"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "rootlens/data/reports/"
            "graphsage_v2_robustness"
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--mlp-hidden",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=7e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=2e-4,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = args.repo_root.resolve()

    graph_dir = resolve_path(
        args.graph_dir,
        repo_root,
    )

    output_dir = resolve_path(
        args.output_dir,
        repo_root,
    )

    try:
        train_path = (
            graph_dir
            / "train_graphs.pt"
        )

        val_path = (
            graph_dir
            / "validation_graphs.pt"
        )

        test_path = (
            graph_dir
            / "test_graphs.pt"
        )

        # Test must exist but is NEVER loaded.
        if not test_path.is_file():
            raise FileNotFoundError(
                "Sealed test artifact missing"
            )

        base_train = load_graphs(
            train_path
        )

        base_val = load_graphs(
            val_path
        )

        validate_no_leakage(
            base_train,
            base_val,
        )

        mean, std = compute_normalization(
            base_train
        )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        tracking_uri = configure_mlflow(
            repo_root
        )

        rows = []
        all_run_predictions = []

        conditions = [
            "real_graph",
            "self_loops_only",
        ]

        for condition in conditions:
            for seed in args.seeds:
                print()
                print(
                    f"Running condition={condition}, "
                    f"seed={seed}"
                )

                with mlflow.start_run(
                    run_name=(
                        f"v2_{condition}_seed_{seed}"
                    )
                ) as mlflow_run:

                    result = run_experiment(
                        seed=seed,
                        condition=condition,
                        base_train=base_train,
                        base_val=base_val,
                        mean=mean,
                        std=std,
                        device=device,
                        epochs=args.epochs,
                        patience=args.patience,
                        batch_size=args.batch_size,
                        hidden_channels=args.hidden_channels,
                        mlp_hidden=args.mlp_hidden,
                        dropout=args.dropout,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                    )

                    mlflow.set_tags(
                        {
                            "project": "RootLensAI",
                            "study": "robustness_and_topology_ablation",
                            "topology_condition": condition,
                            "test_set_evaluated": "false",
                        }
                    )

                    mlflow.log_params(
                        {
                            "seed": seed,
                            "topology_condition": condition,
                            "epochs_max": args.epochs,
                            "patience": args.patience,
                            "batch_size": args.batch_size,
                            "hidden_channels": args.hidden_channels,
                            "mlp_hidden": args.mlp_hidden,
                            "dropout": args.dropout,
                            "learning_rate": args.learning_rate,
                            "weight_decay": args.weight_decay,
                            "device": str(device),
                        }
                    )

                    mlflow.log_metrics(
                        {
                            "best_epoch": float(
                                result["best_epoch"]
                            ),
                            "val_window_accuracy": (
                                result["window_accuracy"]
                            ),
                            "val_window_macro_f1": (
                                result["window_macro_f1"]
                            ),
                            "val_window_weighted_f1": (
                                result["window_weighted_f1"]
                            ),
                            "val_run_accuracy": (
                                result["run_accuracy"]
                            ),
                            "val_run_macro_f1": (
                                result["run_macro_f1"]
                            ),
                            "val_correct_runs": float(
                                result["correct_runs"]
                            ),
                        }
                    )

                    print(
                        f"  best_epoch={result['best_epoch']} | "
                        f"window_macro_f1="
                        f"{result['window_macro_f1']:.4f} | "
                        f"run_accuracy="
                        f"{result['correct_runs']}/"
                        f"{result['total_runs']}"
                    )

                    rows.append(
                        {
                            "mlflow_run_id": (
                                mlflow_run.info.run_id
                            ),
                            "seed": seed,
                            "condition": condition,
                            "best_epoch": (
                                result["best_epoch"]
                            ),
                            "window_accuracy": (
                                result["window_accuracy"]
                            ),
                            "window_macro_f1": (
                                result["window_macro_f1"]
                            ),
                            "window_weighted_f1": (
                                result["window_weighted_f1"]
                            ),
                            "run_accuracy": (
                                result["run_accuracy"]
                            ),
                            "run_macro_f1": (
                                result["run_macro_f1"]
                            ),
                            "correct_runs": (
                                result["correct_runs"]
                            ),
                            "total_runs": (
                                result["total_runs"]
                            ),
                        }
                    )

                    pred_df = (
                        result["run_predictions"]
                        .copy()
                    )

                    pred_df[
                        "seed"
                    ] = seed

                    pred_df[
                        "condition"
                    ] = condition

                    all_run_predictions.append(
                        pred_df
                    )

        results = pd.DataFrame(
            rows
        )

        results_path = (
            output_dir
            / "robustness_results.csv"
        )

        results.to_csv(
            results_path,
            index=False,
        )

        run_predictions = pd.concat(
            all_run_predictions,
            ignore_index=True,
        )

        run_predictions_path = (
            output_dir
            / "run_level_predictions.csv"
        )

        run_predictions.to_csv(
            run_predictions_path,
            index=False,
        )

        summary_rows = []

        for condition, group in results.groupby(
            "condition"
        ):
            summary_rows.append(
                {
                    "condition": condition,
                    "n_seeds": len(group),
                    "window_macro_f1_mean": float(
                        group[
                            "window_macro_f1"
                        ].mean()
                    ),
                    "window_macro_f1_std": float(
                        group[
                            "window_macro_f1"
                        ].std(ddof=0)
                    ),
                    "window_macro_f1_min": float(
                        group[
                            "window_macro_f1"
                        ].min()
                    ),
                    "window_macro_f1_max": float(
                        group[
                            "window_macro_f1"
                        ].max()
                    ),
                    "run_accuracy_mean": float(
                        group[
                            "run_accuracy"
                        ].mean()
                    ),
                    "run_accuracy_min": float(
                        group[
                            "run_accuracy"
                        ].min()
                    ),
                    "perfect_run_localization_seeds": int(
                        (
                            group[
                                "correct_runs"
                            ]
                            == group[
                                "total_runs"
                            ]
                        ).sum()
                    ),
                }
            )

        summary = pd.DataFrame(
            summary_rows
        )

        summary_path = (
            output_dir
            / "robustness_summary.csv"
        )

        summary.to_csv(
            summary_path,
            index=False,
        )

        seed_plot = (
            output_dir
            / "seed_robustness.png"
        )

        save_seed_plot(
            results,
            seed_plot,
        )

        ablation_plot = (
            output_dir
            / "topology_ablation.png"
        )

        save_ablation_plot(
            summary,
            ablation_plot,
        )

        json_summary = {
            "test_set_evaluated": False,
            "seeds": args.seeds,
            "device": str(device),
            "random_forest_reference_macro_f1": 0.9683,
            "conditions": summary.to_dict(
                orient="records"
            ),
        }

        json_path = (
            output_dir
            / "robustness_summary.json"
        )

        with json_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                json_summary,
                f,
                indent=2,
            )

        print()
        print("=" * 96)
        print(
            "RootLens GraphSAGE v2 Robustness + Topology Ablation"
        )
        print("=" * 96)

        for _, row in summary.iterrows():
            print(
                f"{row['condition']:<22} "
                f"macro_f1 mean="
                f"{row['window_macro_f1_mean']:.4f} "
                f"std="
                f"{row['window_macro_f1_std']:.4f} "
                f"min="
                f"{row['window_macro_f1_min']:.4f} "
                f"max="
                f"{row['window_macro_f1_max']:.4f} | "
                f"perfect run seeds="
                f"{int(row['perfect_run_localization_seeds'])}/"
                f"{int(row['n_seeds'])}"
            )

        print()
        print(
            "Random Forest validation macro F1 reference: 0.9683"
        )
        print()
        print(f"Results:          {results_path}")
        print(f"Run predictions:  {run_predictions_path}")
        print(f"Seed plot:        {seed_plot}")
        print(f"Ablation plot:    {ablation_plot}")
        print(f"MLflow:           {tracking_uri}")
        print()
        print("TEST SET REMAINS SEALED.")
        print("=" * 96)
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