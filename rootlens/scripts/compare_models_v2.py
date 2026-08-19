#scripts/compare_models_v2.py
#!/usr/bin/env python3
"""Create and MLflow-log Dataset v2 Phase-1 validation-only comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "rootlens/data/reports/baseline_results_v2"
GNN = ROOT / "rootlens/data/reports/graphsage_v1_dataset_v2"
OUT = ROOT / "rootlens/data/reports/model_comparison_v2_phase1"
DB = ROOT / "rootlens/mlflow.db"
TRACKING_URI = f"sqlite:///{DB.as_posix()}"
EXPERIMENT = "RootLens-RCA-Baselines-v2"
CLASSES = ["healthy", "payment", "cart", "checkout", "product_catalog"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(BASE / "baseline_comparison.csv")
    baseline["display_name"] = baseline["model"].map({
        "dummy_most_frequent": "Dummy Classifier",
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
    })

    gnn_summary = load_json(GNN / "run_summary.json")
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    rows: list[dict] = []
    reports: dict[str, dict] = {}
    for _, row in baseline.iterrows():
        run = client.get_run(str(row["mlflow_run_id"]))
        name = str(row["display_name"])
        rows.append({
            "model": name,
            "mlflow_run_id": row["mlflow_run_id"],
            "validation_accuracy": row["val_accuracy"],
            "validation_macro_f1": row["val_macro_f1"],
            "validation_weighted_f1": row["val_weighted_f1"],
            "training_duration_seconds": row["training_duration_seconds"],
            "parameter_count": np.nan,
        })
        reports[name] = load_json(
            BASE / str(row["model"]) / "classification_report.json"
        )

    gnn_id = str(gnn_summary["mlflow_run_id"])
    gnn_run = client.get_run(gnn_id)
    rows.append({
        "model": "GraphSAGE v1",
        "mlflow_run_id": gnn_id,
        "validation_accuracy": gnn_summary["best_validation_accuracy"],
        "validation_macro_f1": gnn_summary["best_validation_macro_f1"],
        "validation_weighted_f1": gnn_summary["best_validation_weighted_f1"],
        "training_duration_seconds": gnn_run.data.metrics["training_duration_seconds"],
        "parameter_count": int(gnn_run.data.params["parameter_count"]),
    })
    reports["GraphSAGE v1"] = load_json(
        GNN / "validation_classification_report.json"
    )

    comparison = pd.DataFrame(rows).sort_values(
        "validation_macro_f1", ascending=False
    )
    comparison.to_csv(OUT / "validation_model_comparison.csv", index=False)

    colors = ["#64748b", "#2563eb", "#16a34a", "#9333ea"]
    for metric, title, filename in [
        ("validation_accuracy", "Dataset v2 Validation Accuracy", "validation_accuracy_comparison.png"),
        ("validation_macro_f1", "Dataset v2 Validation Macro-F1", "validation_macro_f1_comparison.png"),
    ]:
        ordered = comparison.sort_values(metric)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        bars = ax.barh(ordered["model"], ordered[metric], color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(metric.replace("validation_", "").replace("_", " ").title())
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=180)
        plt.close(fig)

    f1_rows = []
    for model, report in reports.items():
        for label in CLASSES:
            f1_rows.append({
                "model": model,
                "class": label,
                "f1": report[label]["f1-score"],
            })
    f1 = pd.DataFrame(f1_rows)
    f1.to_csv(OUT / "per_class_f1.csv", index=False)
    pivot = f1.pivot(index="class", columns="model", values="f1").reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validation F1")
    ax.set_xlabel("Root-cause class")
    ax.set_title("Dataset v2 Per-Class Validation F1")
    ax.legend(title="Model", loc="lower right")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUT / "per_class_f1_comparison.png", dpi=180)
    plt.close(fig)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="phase1_validation_comparison") as run:
        mlflow.set_tags({
            "dataset_version": "v2",
            "comparison_scope": "validation_only",
            "test_set_evaluated": "false",
        })
        mlflow.log_params({
            "seed": 42,
            "model_count": 4,
            "primary_selection_metric": "validation_macro_f1",
        })
        for path in OUT.iterdir():
            mlflow.log_artifact(str(path), artifact_path="validation_comparison")
        comparison_run_id = run.info.run_id

    metadata = {
        "dataset_version": "v2",
        "scope": "validation_only",
        "sealed_test_evaluated": False,
        "comparison_mlflow_run_id": comparison_run_id,
        "model_run_ids": dict(zip(comparison["model"], comparison["mlflow_run_id"])),
    }
    (OUT / "comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    print(f"comparison_mlflow_run_id={comparison_run_id}")
    print("SEALED TEST EVALUATED: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
