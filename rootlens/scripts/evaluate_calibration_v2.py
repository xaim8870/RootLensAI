#!/usr/bin/env python3
"""Professional calibration evaluation for the frozen RootLens GraphSAGE RCA v2 model.

What this script does
---------------------
1. Loads the exact frozen Dataset v2 model bundle used by live RootLens inference.
2. Evaluates VALIDATION graphs only; the sealed test set is checked for existence
   but is never loaded.
3. Computes:
      - validation accuracy
      - mean predicted confidence
      - accuracy-confidence gap
      - Expected Calibration Error (ECE)
      - Negative Log-Likelihood (NLL)
      - multiclass Brier score
4. Saves:
      - presentation-ready reliability diagram
      - presentation-ready confidence histogram
      - presentation-ready calibration summary figure
      - calibration bins CSV
      - per-window probability CSV
      - JSON metrics summary
5. Logs all metrics and figures to a dedicated MLflow experiment.

Usage
-----
From the repository root:

    python rootlens\\scripts\\evaluate_calibration_v2.py
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import accuracy_score, log_loss
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# ---------------------------------------------------------------------------
# Make repository package imports work when this file is executed directly.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from rootlens.inference.rca_inference import DEFAULT_BUNDLE, RCAInference  # noqa: E402


EXPERIMENT_NAME = "RootLens-RCA-GNN-v2-Calibration"

# Light presentation palette.
COLOR_BLUE = "#8FB8E8"
COLOR_GREEN = "#A7D9B5"
COLOR_PURPLE = "#C1AFE8"
COLOR_ORANGE = "#F3C58F"
COLOR_RED = "#EBA0A0"
COLOR_GRAY = "#CBD3DC"
COLOR_DARK = "#354052"
COLOR_GRID = "#E7EBF0"
COLOR_BACKGROUND = "#FAFBFC"


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def load_graphs(path: Path) -> list[Data]:
    graphs = torch.load(path, weights_only=False)

    if not isinstance(graphs, list) or not graphs:
        raise ValueError(f"Invalid graph artifact: {path}")

    return graphs


def normalize_graphs(
    graphs: list[Data],
    mean: torch.Tensor,
    std: torch.Tensor,
    edge_index: torch.Tensor,
) -> list[Data]:
    """Apply the exact normalization/topology used by frozen live inference."""

    output: list[Data] = []

    mean_cpu = mean.detach().cpu()
    std_cpu = std.detach().cpu()
    edge_index_cpu = edge_index.detach().cpu()

    for graph in graphs:
        g = copy.deepcopy(graph)
        g.x = (g.x.float() - mean_cpu) / std_cpu
        g.edge_index = edge_index_cpu.clone()
        output.append(g)

    return output


def normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    """Renormalize softmax rows to exactly sum to one in float64.

    PyTorch float32 softmax can produce row sums such as 0.99999997 or
    1.00000003 after conversion to NumPy. sklearn's log_loss may warn about
    this even though the probabilities are mathematically valid.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    row_sums = probabilities.sum(axis=1, keepdims=True)

    if np.any(row_sums <= 0):
        raise ValueError("Probability row with non-positive sum encountered")

    return probabilities / row_sums


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    engine: RCAInference,
    graphs: list[Data],
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    run_ids: list[str] = []
    timestamps: list[str] = []

    engine.model.eval()

    for batch in loader:
        batch = batch.to(engine.device)

        logits = engine.model(
            batch.x,
            batch.edge_index,
            batch.batch,
        )

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        y_true.extend(batch.y.view(-1).cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        probabilities.extend(probs.cpu().tolist())
        run_ids.extend([str(value) for value in batch.run_id])
        timestamps.extend([str(value) for value in batch.timestamp])

    probability_array = normalize_probability_rows(
        np.asarray(probabilities, dtype=np.float64)
    )

    return {
        "y_true": np.asarray(y_true, dtype=np.int64),
        "y_pred": np.asarray(y_pred, dtype=np.int64),
        "probabilities": probability_array,
        "run_ids": run_ids,
        "timestamps": timestamps,
    }


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int,
) -> tuple[float, pd.DataFrame]:
    """Top-label Expected Calibration Error with equal-width bins."""

    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0

    for index in range(n_bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])

        if index == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        count = int(mask.sum())

        if count == 0:
            rows.append(
                {
                    "bin_index": index,
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "mean_confidence": np.nan,
                    "accuracy": np.nan,
                    "absolute_gap": np.nan,
                }
            )
            continue

        mean_confidence = float(confidences[mask].mean())
        observed_accuracy = float(correctness[mask].mean())
        absolute_gap = abs(observed_accuracy - mean_confidence)

        ece += (count / len(confidences)) * absolute_gap

        rows.append(
            {
                "bin_index": index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": observed_accuracy,
                "absolute_gap": absolute_gap,
            }
        )

    return float(ece), pd.DataFrame(rows)


def multiclass_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Mean squared distance between predicted probabilities and one-hot truth."""

    one_hot = np.zeros_like(probabilities, dtype=np.float64)
    one_hot[np.arange(len(y_true)), y_true] = 1.0

    return float(
        np.mean(
            np.sum(
                (probabilities - one_hot) ** 2,
                axis=1,
            )
        )
    )


# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(COLOR_BACKGROUND)
    ax.grid(
        axis="y",
        color=COLOR_GRID,
        linewidth=0.8,
        alpha=0.9,
    )
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_GRAY)

    ax.tick_params(colors=COLOR_DARK)
    ax.xaxis.label.set_color(COLOR_DARK)
    ax.yaxis.label.set_color(COLOR_DARK)
    ax.title.set_color(COLOR_DARK)


# ---------------------------------------------------------------------------
# Presentation-ready plots
# ---------------------------------------------------------------------------

def save_reliability_diagram(
    bins: pd.DataFrame,
    ece: float,
    path: Path,
) -> None:
    """Reliability diagram: predicted confidence vs observed correctness."""

    valid = bins.dropna(
        subset=["mean_confidence", "accuracy"]
    )

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(COLOR_BACKGROUND)

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.7,
        color=COLOR_GRAY,
        label="Perfect calibration",
    )

    if not valid.empty:
        ax.plot(
            valid["mean_confidence"],
            valid["accuracy"],
            marker="o",
            markersize=8,
            linewidth=2.5,
            color=COLOR_BLUE,
            markerfacecolor="white",
            markeredgewidth=2,
            label="GraphSAGE v2",
        )

        for _, row in valid.iterrows():
            ax.vlines(
                row["mean_confidence"],
                min(row["mean_confidence"], row["accuracy"]),
                max(row["mean_confidence"], row["accuracy"]),
                color=COLOR_PURPLE,
                alpha=0.5,
                linewidth=2,
            )

    ax.set_xlim(0.45, 1.01)
    ax.set_ylim(0.45, 1.01)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.set_xlabel("Mean predicted confidence", fontsize=11)
    ax.set_ylabel("Observed accuracy", fontsize=11)
    ax.set_title(
        "RootLens GraphSAGE v2 — Reliability Diagram",
        fontsize=15,
        pad=16,
        weight="semibold",
    )

    ax.text(
        0.03,
        0.95,
        f"ECE = {ece * 100:.2f}%",
        transform=ax.transAxes,
        fontsize=12,
        weight="semibold",
        color=COLOR_DARK,
        va="top",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#F2F6FB",
            "edgecolor": COLOR_BLUE,
            "alpha": 0.95,
        },
    )

    ax.grid(
        color=COLOR_GRID,
        linewidth=0.8,
        alpha=0.9,
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_GRAY)

    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def save_confidence_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    path: Path,
) -> None:
    """Confidence distribution split into correct vs incorrect predictions."""

    confidences = probabilities.max(axis=1)
    correct_mask = y_true == y_pred

    bins = np.linspace(0.0, 1.0, 11)

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    fig.patch.set_facecolor("white")

    ax.hist(
        confidences[correct_mask],
        bins=bins,
        color=COLOR_GREEN,
        alpha=0.88,
        label="Correct predictions",
        edgecolor="white",
    )

    if np.any(~correct_mask):
        ax.hist(
            confidences[~correct_mask],
            bins=bins,
            color=COLOR_RED,
            alpha=0.88,
            label="Incorrect predictions",
            edgecolor="white",
        )

    ax.axvline(
        confidences.mean(),
        color=COLOR_BLUE,
        linestyle="--",
        linewidth=2,
        label=f"Mean confidence {confidences.mean() * 100:.2f}%",
    )

    ax.set_xlim(0.45, 1.01)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Predicted-class confidence", fontsize=11)
    ax.set_ylabel("Validation windows", fontsize=11)
    ax.set_title(
        "RootLens GraphSAGE v2 — Confidence Distribution",
        fontsize=15,
        pad=16,
        weight="semibold",
    )

    style_axis(ax)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def save_calibration_summary(
    accuracy: float,
    mean_confidence: float,
    confidence_gap: float,
    ece: float,
    nll: float,
    brier: float,
    path: Path,
) -> None:
    """Single presentation figure summarizing the calibration evaluation."""

    metric_labels = [
        "Validation\nAccuracy",
        "Mean\nConfidence",
        "Accuracy–Confidence\nGap",
        "ECE",
    ]

    metric_values = [
        accuracy * 100,
        mean_confidence * 100,
        confidence_gap * 100,
        ece * 100,
    ]

    metric_colors = [
        COLOR_GREEN,
        COLOR_BLUE,
        COLOR_ORANGE,
        COLOR_PURPLE,
    ]

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    fig.patch.set_facecolor("white")

    bars = ax.bar(
        metric_labels,
        metric_values,
        color=metric_colors,
        width=0.62,
        edgecolor="white",
        linewidth=1.2,
    )

    ax.set_ylabel("Percent", fontsize=11)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(PercentFormatter(100))

    ax.set_title(
        "RootLens GraphSAGE v2 — Confidence Calibration Evaluation",
        fontsize=16,
        pad=18,
        weight="semibold",
    )

    for bar, value in zip(bars, metric_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="semibold",
            color=COLOR_DARK,
        )

    style_axis(ax)

    ax.text(
        0.5,
        -0.19,
        f"NLL = {nll:.4f}     •     Multiclass Brier score = {brier:.4f}     •     Lower is better",
        transform=ax.transAxes,
        ha="center",
        fontsize=10.5,
        color=COLOR_DARK,
    )

    ax.text(
        0.5,
        -0.27,
        "Validation-only calibration evaluation · sealed test set not evaluated",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.5,
        color="#758092",
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=240,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

def configure_mlflow(repo_root: Path) -> str:
    db_path = (
        repo_root
        / "rootlens"
        / "mlflow.db"
    ).resolve()

    tracking_uri = f"sqlite:///{db_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    return tracking_uri


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Professional calibration evaluation for the frozen "
            "RootLens GraphSAGE RCA v2 model."
        )
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
    )

    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE,
    )

    parser.add_argument(
        "--validation-graphs",
        type=Path,
        default=Path(
            "rootlens/data/processed/dataset_v2/graphs/"
            "validation_graphs.pt"
        ),
    )

    parser.add_argument(
        "--test-graphs",
        type=Path,
        default=Path(
            "rootlens/data/processed/dataset_v2/graphs/"
            "test_graphs.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "rootlens/data/reports/"
            "graphsage_rca_v2_calibration"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=10,
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    repo_root = args.repo_root.resolve()

    bundle_path = resolve_path(
        args.bundle,
        repo_root,
    )

    validation_path = resolve_path(
        args.validation_graphs,
        repo_root,
    )

    test_path = resolve_path(
        args.test_graphs,
        repo_root,
    )

    output_dir = resolve_path(
        args.output_dir,
        repo_root,
    )

    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"Frozen model bundle not found: {bundle_path}"
        )

    if not validation_path.is_file():
        raise FileNotFoundError(
            f"Validation graphs not found: {validation_path}"
        )

    if not test_path.is_file():
        raise FileNotFoundError(
            "Sealed test artifact is missing. "
            "The calibration script will not continue."
        )

    # -----------------------------------------------------------------------
    # Verify the canonical frozen bundle.
    # -----------------------------------------------------------------------

    bundle = json.loads(
        bundle_path.read_text(encoding="utf-8")
    )

    if bundle.get("dataset_version") != "v2":
        raise ValueError(
            "Calibration must target the frozen Dataset v2 model. "
            f"Bundle reports {bundle.get('dataset_version')!r}."
        )

    if bundle.get("sealed_test_evaluated") is not False:
        raise ValueError(
            "Frozen bundle no longer reports the test set as sealed."
        )

    # -----------------------------------------------------------------------
    # Load exact live inference model.
    # -----------------------------------------------------------------------

    engine = RCAInference(
        bundle_path=bundle_path,
        device="cpu",
    )

    validation_graphs = load_graphs(
        validation_path
    )

    validation_graphs = normalize_graphs(
        validation_graphs,
        engine.mean,
        engine.std,
        engine.edge_index,
    )

    result = collect_predictions(
        engine=engine,
        graphs=validation_graphs,
        batch_size=args.batch_size,
    )

    y_true = result["y_true"]
    y_pred = result["y_pred"]
    probabilities = result["probabilities"]

    # -----------------------------------------------------------------------
    # Metrics.
    # -----------------------------------------------------------------------

    accuracy = float(
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    mean_confidence = float(
        probabilities.max(axis=1).mean()
    )

    confidence_gap = abs(
        mean_confidence - accuracy
    )

    ece, calibration_bins = expected_calibration_error(
        y_true=y_true,
        probabilities=probabilities,
        n_bins=args.bins,
    )

    nll = float(
        log_loss(
            y_true,
            probabilities,
            labels=list(
                range(
                    probabilities.shape[1]
                )
            ),
        )
    )

    brier = multiclass_brier_score(
        y_true=y_true,
        probabilities=probabilities,
    )

    # -----------------------------------------------------------------------
    # Output files.
    # -----------------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = {
        "validation_accuracy": accuracy,
        "validation_mean_confidence": mean_confidence,
        "validation_accuracy_confidence_gap": confidence_gap,
        "validation_ece": ece,
        "validation_nll": nll,
        "validation_brier_score": brier,
        "calibration_bins": int(args.bins),
        "validation_windows": int(len(y_true)),
        "sealed_test_evaluated": False,
        "canonical_model_mlflow_run_id": bundle.get(
            "mlflow_run_id"
        ),
        "dataset_version": bundle.get(
            "dataset_version"
        ),
        "probability_source": "raw_softmax",
    }

    metrics_path = (
        output_dir
        / "validation_calibration_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    bins_path = (
        output_dir
        / "validation_calibration_bins.csv"
    )

    calibration_bins.to_csv(
        bins_path,
        index=False,
    )

    index_to_class = {
        int(index): label
        for label, index in engine.class_to_index.items()
    }

    prediction_data: dict[str, Any] = {
        "run_id": result["run_ids"],
        "timestamp": result["timestamps"],
        "true_label": [
            index_to_class[int(index)]
            for index in y_true
        ],
        "predicted_label": [
            index_to_class[int(index)]
            for index in y_pred
        ],
        "confidence": probabilities.max(axis=1),
        "correct": y_true == y_pred,
    }

    for class_index in range(
        probabilities.shape[1]
    ):
        label = index_to_class[
            class_index
        ]

        prediction_data[
            f"prob_{label}"
        ] = probabilities[
            :,
            class_index,
        ]

    predictions_path = (
        output_dir
        / "validation_calibration_predictions.csv"
    )

    pd.DataFrame(
        prediction_data
    ).to_csv(
        predictions_path,
        index=False,
    )

    reliability_path = (
        output_dir
        / "validation_reliability_diagram.png"
    )

    confidence_path = (
        output_dir
        / "validation_confidence_histogram.png"
    )

    summary_figure_path = (
        output_dir
        / "validation_calibration_summary.png"
    )

    save_reliability_diagram(
        calibration_bins,
        ece,
        reliability_path,
    )

    save_confidence_histogram(
        y_true,
        y_pred,
        probabilities,
        confidence_path,
    )

    save_calibration_summary(
        accuracy=accuracy,
        mean_confidence=mean_confidence,
        confidence_gap=confidence_gap,
        ece=ece,
        nll=nll,
        brier=brier,
        path=summary_figure_path,
    )

    # -----------------------------------------------------------------------
    # MLflow.
    # -----------------------------------------------------------------------

    tracking_uri = configure_mlflow(
        repo_root
    )

    with mlflow.start_run(
        run_name=(
            "graphsage_rca_v2_"
            "professional_calibration_validation"
        )
    ) as run:

        mlflow.set_tags(
            {
                "project": "RootLensAI",
                "task": "confidence_calibration_evaluation",
                "model_family": "GraphSAGE",
                "dataset_version": "v2",
                "evaluation_scope": "validation_only",
                "calibration_scope": "raw_softmax_validation_only",
                "temperature_scaling_applied": "false",
                "test_set_evaluated": "false",
                "canonical_model_run_id": str(
                    bundle.get(
                        "mlflow_run_id",
                        "unknown",
                    )
                ),
                "presentation_artifacts": "true",
            }
        )

        mlflow.log_params(
            {
                "canonical_model_name": str(
                    bundle.get(
                        "model_name",
                        "unknown",
                    )
                ),
                "canonical_model_mlflow_run_id": str(
                    bundle.get(
                        "mlflow_run_id",
                        "unknown",
                    )
                ),
                "calibration_bins": args.bins,
                "batch_size": args.batch_size,
                "probability_source": "raw_softmax",
                "sealed_test_policy": (
                    "existence_checked_not_loaded"
                ),
                "promql_window": str(
                    bundle.get(
                        "promql_window",
                        "unknown",
                    )
                ),
            }
        )

        mlflow.log_metrics(
            {
                "val_accuracy": accuracy,
                "val_mean_confidence": mean_confidence,
                "val_accuracy_confidence_gap": confidence_gap,
                "val_ece": ece,
                "val_nll": nll,
                "val_brier_score": brier,
            }
        )

        for artifact in [
            metrics_path,
            bins_path,
            predictions_path,
            reliability_path,
            confidence_path,
            summary_figure_path,
        ]:
            mlflow.log_artifact(
                str(artifact),
                artifact_path="calibration",
            )

        calibration_run_id = (
            run.info.run_id
        )

    # -----------------------------------------------------------------------
    # Terminal report.
    # -----------------------------------------------------------------------

    print()
    print("=" * 92)
    print(
        "RootLens GraphSAGE RCA v2 — "
        "Professional Validation Calibration Evaluation"
    )
    print("=" * 92)

    print(
        f"Canonical model run:            "
        f"{bundle.get('mlflow_run_id')}"
    )

    print(
        f"Calibration MLflow run:         "
        f"{calibration_run_id}"
    )

    print(
        f"Dataset version:                "
        f"{bundle.get('dataset_version')}"
    )

    print(
        f"Validation windows:             "
        f"{len(y_true)}"
    )

    print()

    print(
        f"Validation accuracy:            "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Mean confidence:                "
        f"{mean_confidence:.4f} "
        f"({mean_confidence * 100:.2f}%)"
    )

    print(
        f"Accuracy-confidence gap:        "
        f"{confidence_gap:.4f} "
        f"({confidence_gap * 100:.2f} pp)"
    )

    print(
        f"Expected Calibration Error:     "
        f"{ece:.4f} "
        f"({ece * 100:.2f}%)"
    )

    print(
        f"Negative Log-Likelihood:        "
        f"{nll:.4f}"
    )

    print(
        f"Multiclass Brier score:         "
        f"{brier:.4f}"
    )

    print()
    print("PRESENTATION ARTIFACTS")
    print("-" * 92)

    print(
        f"Calibration summary:            "
        f"{summary_figure_path}"
    )

    print(
        f"Reliability diagram:            "
        f"{reliability_path}"
    )

    print(
        f"Confidence distribution:        "
        f"{confidence_path}"
    )

    print()
    print("SEALED TEST EVALUATED: NO")
    print(
        f"MLflow tracking URI:            "
        f"{tracking_uri}"
    )

    print("=" * 92)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )