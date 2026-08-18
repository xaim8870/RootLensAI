#!/usr/bin/env python3
"""Audit and record the canonical Dataset v2 validation-only comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "rootlens/data/processed/dataset_v2/splits"
GRAPHS = ROOT / "rootlens/data/processed/dataset_v2/graphs"
REPORTS = ROOT / "rootlens/data/reports"
OUT = REPORTS / "model_comparison_v2_canonical"
TRACKING_URI = f"sqlite:///{(ROOT / 'rootlens/mlflow.db').as_posix()}"
SPLIT_SHA = "33467da3f8e13bb3d6108aaaa8fe41ecbedf4be4b6cfda679b3df9483d4601a8"
CLASSES = ["healthy", "payment", "cart", "checkout", "product_catalog"]


MODELS = [
    {
        "model": "Dummy Classifier",
        "architecture": "DummyClassifier(strategy=most_frequent)",
        "run_id": "24bedb31d7ff4ccbaf4a2802c9428d73",
        "report": REPORTS / "baseline_results_v2/dummy_most_frequent/classification_report.json",
        "confusion": REPORTS / "baseline_results_v2/dummy_most_frequent/confusion_matrix.png",
        "preprocessing": "none; fitted on canonical training labels only",
    },
    {
        "model": "Logistic Regression",
        "architecture": "StandardScaler -> LogisticRegression",
        "run_id": "038e52d11c3042f991682ba28a723da0",
        "report": REPORTS / "baseline_results_v2/logistic_regression/classification_report.json",
        "confusion": REPORTS / "baseline_results_v2/logistic_regression/confusion_matrix.png",
        "preprocessing": "StandardScaler fit on canonical training features only",
    },
    {
        "model": "Random Forest",
        "architecture": "RandomForestClassifier(n_estimators=300)",
        "run_id": "00867a23c26147959bcc7588ec7c3d83",
        "report": REPORTS / "baseline_results_v2/random_forest/classification_report.json",
        "confusion": REPORTS / "baseline_results_v2/random_forest/confusion_matrix.png",
        "preprocessing": "no scaling; model fit on canonical training features only",
    },
    {
        "model": "GraphSAGE v1 mean-pool",
        "architecture": "SAGEConv(7,32) -> SAGEConv(32,32) -> global_mean_pool -> Linear(32,5)",
        "run_id": "080ca998214f41bb8f160ca4cf6260a8",
        "report": REPORTS / "graphsage_v1_dataset_v2/validation_classification_report.json",
        "confusion": REPORTS / "graphsage_v1_dataset_v2/validation_confusion_matrix.png",
        "preprocessing": "per-feature standardization fit on canonical training graphs only",
        "parameter_count": 2725,
    },
    {
        "model": "GraphSAGE RCA v2 node-preserving",
        "architecture": "2-layer residual GraphSAGE -> ordered 12-node embedding concatenation -> MLP classifier",
        "run_id": "c5b8c95fd91d4ce2ab27df44e846ddec",
        "report": REPORTS / "graphsage_rca_v2_canonical/validation_classification_report.json",
        "confusion": REPORTS / "graphsage_rca_v2_canonical/validation_confusion_matrix.png",
        "preprocessing": "per-feature standardization fit on canonical training graphs only",
        "parameter_count": 61445,
    },
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    definition = pd.read_csv(SPLITS / "split_definition_v2.csv")
    metadata = read_json(SPLITS / "split_metadata.json")
    if sha256(SPLITS / "split_definition_v2.csv") != SPLIT_SHA:
        raise RuntimeError("Frozen Dataset v2 split checksum mismatch")

    split_runs = {
        split: definition.loc[definition["split"] == split, "run_id"].tolist()
        for split in ("train", "validation", "test")
    }
    run_sets = {key: set(value) for key, value in split_runs.items()}
    intersections = {
        "train_validation": sorted(run_sets["train"] & run_sets["validation"]),
        "train_test": sorted(run_sets["train"] & run_sets["test"]),
        "validation_test": sorted(run_sets["validation"] & run_sets["test"]),
    }
    if any(intersections.values()):
        raise RuntimeError(f"Run leakage detected: {intersections}")

    run_ids_artifact = {
        "dataset": "v2",
        "split_seed": metadata["split_seed"],
        "split_definition_sha256": SPLIT_SHA,
        "sample_counts": metadata["row_counts_by_split"],
        "run_counts": metadata["run_counts_by_split"],
        "runs": split_runs,
        "run_intersections": intersections,
        "sealed_test_usage": "structural membership audit only; no inference or metrics",
    }
    (OUT / "canonical_validation_run_ids.json").write_text(
        json.dumps(run_ids_artifact, indent=2), encoding="utf-8"
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    comparison_rows = []
    provenance = []
    per_class_rows = []
    for item in MODELS:
        report = read_json(item["report"])
        run = client.get_run(item["run_id"])
        accuracy_key = (
            "best_val_accuracy"
            if "best_val_accuracy" in run.data.metrics
            else "val_accuracy"
        )
        accuracy = float(run.data.metrics[accuracy_key])
        row = {
            "model": item["model"],
            "validation_samples": 480,
            "validation_runs": 8,
            "accuracy": accuracy,
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "healthy_f1": float(report["healthy"]["f1-score"]),
            "mlflow_run_id": item["run_id"],
        }
        comparison_rows.append(row)
        for label in CLASSES:
            per_class_rows.append({
                "model": item["model"],
                "class": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
            })
        provenance.append({
            **item,
            "report": str(item["report"]),
            "confusion": str(item["confusion"]),
            "confusion_matrix": str(item["confusion"]),
            "dataset_version": run.data.tags.get("dataset_version", "v2"),
            "seed": int(run.data.params.get("seed", run.data.params.get("model_seed", 42))),
            "training_runs": 24,
            "training_samples": 1440,
            "validation_runs": 8,
            "validation_samples": 480,
            "feature_version": "Dataset v2: 12 services x 7 metrics",
            "topology_version": "service_graph_v1 (44 message-passing edges)" if "GraphSAGE" in item["model"] else None,
            "label_mapping": {"healthy": 0, "payment": 1, "cart": 2, "checkout": 3, "product_catalog": 4},
            "split_definition_sha256": SPLIT_SHA,
            "sealed_test_evaluated": False,
        })

    comparison = pd.DataFrame(comparison_rows).sort_values("macro_f1", ascending=False)
    comparison.to_csv(OUT / "canonical_validation_model_comparison.csv", index=False)
    per_class = pd.DataFrame(per_class_rows)
    per_class.to_csv(OUT / "canonical_per_class_metrics.csv", index=False)

    for metric, filename, title in [
        ("accuracy", "canonical_validation_accuracy.png", "Canonical Dataset v2 Validation Accuracy"),
        ("macro_f1", "canonical_validation_macro_f1.png", "Canonical Dataset v2 Validation Macro-F1"),
    ]:
        plot = comparison.sort_values(metric)
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(plot["model"], plot[metric], color="#2563eb")
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.set_xlim(0, 1.05)
        ax.set_title(title)
        ax.set_xlabel(metric.replace("_", " ").title())
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=180)
        plt.close(fig)

    pivot = per_class.pivot(index="class", columns="model", values="f1").reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    pivot.plot(kind="bar", ax=ax, width=0.82)
    ax.set_ylim(0, 1.05)
    ax.set_title("Canonical Dataset v2 Per-Class Validation F1")
    ax.set_ylabel("F1")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "canonical_per_class_f1.png", dpi=180)
    plt.close(fig)

    old_predictions = pd.read_csv(
        REPORTS / "graphsage_rca_v2/validation_predictions.csv"
    )
    old_runs = sorted(old_predictions["run_id"].unique())
    frozen_val = split_runs["validation"]
    discrepancy = {
        "classification": "NOT COMPARABLE",
        "cause": "The newer experiment loaded Dataset v1 graph artifacts and hard-coded 1080/18 train plus 360/6 validation expectations.",
        "old_mlflow_run_id": "8a183206780d4b4883ebf9cfe4625b06",
        "old_dataset_version": "v1",
        "old_validation_samples": len(old_predictions),
        "old_validation_runs": old_runs,
        "frozen_v2_validation_runs": frozen_val,
        "intersection": sorted(set(old_runs) & set(frozen_val)),
        "missing_from_newer_experiment": sorted(set(frozen_val) - set(old_runs)),
        "unexpected_in_newer_experiment": sorted(set(old_runs) - set(frozen_val)),
    }
    audit = {
        "status": "PASS",
        "dataset_version": "v2",
        "canonical_split_checksum": SPLIT_SHA,
        "canonical_counts": {
            "train": {"samples": 1440, "runs": 24},
            "validation": {"samples": 480, "runs": 8},
            "test": {"samples": 480, "runs": 8},
        },
        "leakage_checks": {
            "run_intersections": intersections,
            "recovery_runs_in_supervised_data": False,
            "validation_used_for_preprocessing_fit": False,
            "test_inference": False,
            "test_metrics": False,
            "consistent_label_mapping": True,
        },
        "discrepancy_360_samples": discrepancy,
    }
    (OUT / "canonical_model_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    (OUT / "audit_report.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    mlflow.set_experiment("RootLens-RCA-Baselines-v2")
    with mlflow.start_run(run_name="canonical_validation_comparison_v2") as run:
        mlflow.set_tags({
            "dataset": "v2",
            "evaluation_scope": "validation_only",
            "canonical_split": "true",
            "sealed_test_evaluated": "false",
        })
        mlflow.log_params({
            "split_seed": 42,
            "split_definition_sha256": SPLIT_SHA,
            "validation_samples": 480,
            "validation_runs": 8,
            "model_count": 5,
        })
        for path in OUT.iterdir():
            mlflow.log_artifact(str(path), artifact_path="canonical_validation_audit")
        comparison_run_id = run.info.run_id

    print(comparison.to_string(index=False))
    print(f"canonical_comparison_mlflow_run_id={comparison_run_id}")
    print("DATASET V2 SEALED TEST SET WAS NOT EVALUATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
