#!/usr/bin/env python3
"""
Train RootLens Dataset v1 baseline RCA models with MLflow tracking.

Baselines:
    1. DummyClassifier
    2. LogisticRegression + StandardScaler
    3. RandomForestClassifier

IMPORTANT RESEARCH POLICY:
    - Train using train.csv only.
    - Compare models using validation.csv.
    - test.csv is intentionally NOT evaluated here.
    - After selecting the best model, run a separate final test script once.

MLflow tracks:
    - dataset version / SHA256
    - Git commit
    - split version / seed
    - feature set
    - model name / hyperparameters
    - validation accuracy
    - validation macro F1
    - validation weighted F1
    - per-class precision / recall / F1 / support
    - confusion matrix
    - classification report
    - trained sklearn model artifact
    - input example + model signature

Run from D:\RootLensAI:

    python rootlens\scripts\train_baselines_mlflow.py

Then open MLflow UI with:

    mlflow ui --backend-store-uri sqlite:///rootlens/mlflow.db --port 5000

and browse:

    http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_NAME = "RootLens-RCA-Baselines-v1"

TARGET_COLUMN = "root_cause_service"

NON_FEATURE_COLUMNS = {
    "run_id",
    "split",
    "run_role",
    "pair_id",
    "condition",
    "root_cause_service",
    "fault_type",
    "fault_family",
    "is_fault",
    "timestamp",
}

EXPECTED_FEATURE_COUNT = 84

CLASS_ORDER = [
    "healthy",
    "payment",
    "cart",
    "checkout",
    "product_catalog",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns = [
        col for col in df.columns
        if col not in NON_FEATURE_COLUMNS
    ]

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} feature columns, "
            f"found {len(feature_columns)}"
        )

    return feature_columns


def validate_split_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    expected_train_rows: int,
    expected_val_rows: int,
    expected_train_runs: int,
    expected_val_runs: int,
) -> None:
    if len(train_df) != expected_train_rows:
        raise ValueError(
            f"Expected {expected_train_rows} training rows, found {len(train_df)}"
        )

    if len(val_df) != expected_val_rows:
        raise ValueError(
            f"Expected {expected_val_rows} validation rows, found {len(val_df)}"
        )

    if train_df["run_id"].nunique() != expected_train_runs:
        raise ValueError(
            f"Expected {expected_train_runs} training runs, found "
            f"{train_df['run_id'].nunique()}"
        )

    if val_df["run_id"].nunique() != expected_val_runs:
        raise ValueError(
            f"Expected {expected_val_runs} validation runs, found "
            f"{val_df['run_id'].nunique()}"
        )

    train_runs = set(train_df["run_id"])
    val_runs = set(val_df["run_id"])

    overlap = train_runs & val_runs
    if overlap:
        raise ValueError(
            f"RUN LEAKAGE between train and validation: {sorted(overlap)}"
        )

    for name, frame in [
        ("train", train_df),
        ("validation", val_df),
    ]:
        if frame[feature_columns].isna().any().any():
            raise ValueError(
                f"{name} feature matrix contains NaN values"
            )

        observed_classes = set(
            frame[TARGET_COLUMN].astype(str).unique()
        )

        if observed_classes != set(CLASS_ORDER):
            raise ValueError(
                f"{name}: class mismatch. "
                f"Observed={sorted(observed_classes)}"
            )


def build_models(seed: int) -> dict[str, Any]:
    """
    Three intentionally simple research baselines.

    Dummy:
        Measures trivial majority-class performance.

    Logistic Regression:
        Tests whether RCA classes are linearly separable after scaling.

    Random Forest:
        Tests nonlinear interactions / thresholds in tabular telemetry.
    """

    models: dict[str, Any] = {}

    models["dummy_most_frequent"] = DummyClassifier(
        strategy="most_frequent"
    )

    models["logistic_regression"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )

    models["random_forest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )

    return models


def model_params_for_logging(
    model_name: str,
    model: Any,
) -> dict[str, Any]:
    if model_name == "dummy_most_frequent":
        return {
            "strategy": model.strategy,
        }

    if model_name == "logistic_regression":
        clf = model.named_steps["classifier"]

        return {
            "scaler": "StandardScaler",
            "penalty": clf.penalty,
            "C": clf.C,
            "solver": clf.solver,
            "max_iter": clf.max_iter,
            "class_weight": str(clf.class_weight),
            "random_state": clf.random_state,
        }

    if model_name == "random_forest":
        return {
            "n_estimators": model.n_estimators,
            "max_depth": str(model.max_depth),
            "min_samples_split": model.min_samples_split,
            "min_samples_leaf": model.min_samples_leaf,
            "max_features": str(model.max_features),
            "class_weight": str(model.class_weight),
            "random_state": model.random_state,
            "n_jobs": model.n_jobs,
        }

    return {}


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
    model_name: str,
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
        f"RootLens Validation Confusion Matrix\n{model_name}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
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
        json.dump(report, f, indent=2)

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(csv_path)

    return report


def log_per_class_metrics(
    report: dict[str, Any],
) -> None:
    for label in CLASS_ORDER:
        values = report[label]

        safe_label = label.replace("-", "_")

        mlflow.log_metric(
            f"val_precision_{safe_label}",
            float(values["precision"]),
        )
        mlflow.log_metric(
            f"val_recall_{safe_label}",
            float(values["recall"]),
        )
        mlflow.log_metric(
            f"val_f1_{safe_label}",
            float(values["f1-score"]),
        )
        mlflow.log_metric(
            f"val_support_{safe_label}",
            float(values["support"]),
        )


def configure_mlflow(
    repo_root: Path,
    experiment_name: str = EXPERIMENT_NAME,
) -> tuple[str, Path]:
    """
    Use a local SQLite MLflow backend.

    On Windows, sqlite:///D:/... is a valid SQLAlchemy-style SQLite URI.
    """
    db_path = (
        repo_root
        / "rootlens"
        / "mlflow.db"
    ).resolve()

    tracking_uri = f"sqlite:///{db_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    return tracking_uri, db_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RootLens RCA baselines with MLflow."
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="RootLens repository root.",
    )

    parser.add_argument(
        "--train",
        type=Path,
        default=Path(
            "rootlens/data/processed/dataset_v1/splits/train.csv"
        ),
        help="Training split CSV.",
    )

    parser.add_argument(
        "--validation",
        type=Path,
        default=Path(
            "rootlens/data/processed/dataset_v1/splits/validation.csv"
        ),
        help="Validation split CSV.",
    )

    parser.add_argument(
        "--dataset-metadata",
        type=Path,
        default=Path(
            "rootlens/data/processed/dataset_v1/"
            "processed_dataset_v1_metadata.json"
        ),
        help="Processed dataset metadata JSON.",
    )

    parser.add_argument(
        "--split-metadata",
        type=Path,
        default=Path(
            "rootlens/data/processed/dataset_v1/splits/"
            "split_metadata.json"
        ),
        help="Grouped split metadata JSON.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "rootlens/data/reports/baseline_results_v1"
        ),
        help="Local evaluation artifact directory.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Model random seed.",
    )

    parser.add_argument(
        "--experiment-name",
        default=EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("rootlens/data/manifests/dataset_v1.yaml"),
        help="Frozen dataset manifest recorded as provenance.",
    )

    return parser.parse_args()


def resolve_path(
    path: Path,
    repo_root: Path,
) -> Path:
    if not path.is_absolute():
        path = repo_root / path

    return path.resolve()


def main() -> int:
    args = parse_args()

    repo_root = args.repo_root.resolve()

    train_path = resolve_path(args.train, repo_root)
    val_path = resolve_path(args.validation, repo_root)
    dataset_metadata_path = resolve_path(
        args.dataset_metadata,
        repo_root,
    )
    split_metadata_path = resolve_path(
        args.split_metadata,
        repo_root,
    )
    manifest_path = resolve_path(args.manifest, repo_root)
    output_dir = resolve_path(
        args.output_dir,
        repo_root,
    )

    try:
        # --------------------------------------------------------------
        # Load frozen inputs
        # --------------------------------------------------------------

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        dataset_metadata = load_json(
            dataset_metadata_path
        )
        split_metadata = load_json(
            split_metadata_path
        )

        feature_columns = get_feature_columns(train_df)

        if get_feature_columns(val_df) != feature_columns:
            raise ValueError(
                "Train and validation feature columns differ"
            )

        validate_split_frames(
            train_df=train_df,
            val_df=val_df,
            feature_columns=feature_columns,
            expected_train_rows=int(split_metadata["row_counts_by_split"]["train"]),
            expected_val_rows=int(split_metadata["row_counts_by_split"]["validation"]),
            expected_train_runs=int(split_metadata["run_counts_by_split"]["train"]),
            expected_val_runs=int(split_metadata["run_counts_by_split"]["validation"]),
        )

        X_train = train_df[feature_columns]
        y_train = train_df[TARGET_COLUMN].astype(str)

        X_val = val_df[feature_columns]
        y_val = val_df[TARGET_COLUMN].astype(str)

        # --------------------------------------------------------------
        # MLflow setup
        # --------------------------------------------------------------

        tracking_uri, db_path = configure_mlflow(
            repo_root=repo_root,
            experiment_name=args.experiment_name,
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        models = build_models(seed=args.seed)

        comparison_rows: list[dict[str, Any]] = []

        # MLflow dataset lineage.
        train_dataset = mlflow.data.from_pandas(
            train_df,
            source=str(train_path),
            name="rootlens_dataset_v1_train",
            targets=TARGET_COLUMN,
        )

        val_dataset = mlflow.data.from_pandas(
            val_df,
            source=str(val_path),
            name="rootlens_dataset_v1_validation",
            targets=TARGET_COLUMN,
        )

        # --------------------------------------------------------------
        # Train / evaluate each baseline
        # --------------------------------------------------------------

        for model_name, model in models.items():

            model_output_dir = output_dir / model_name
            model_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            with mlflow.start_run(
                run_name=model_name
            ) as run:

                # ------------------------
                # Provenance / tags
                # ------------------------

                mlflow.set_tags(
                    {
                        "project": "RootLensAI",
                        "task": "root_cause_service_classification",
                        "dataset_version": str(
                            dataset_metadata.get(
                                "dataset_version",
                                "v1",
                            )
                        ),
                        "split_version": str(
                            split_metadata.get(
                                "split_version",
                                "v1",
                            )
                        ),
                        "sample_unit": str(
                            dataset_metadata.get(
                                "sample_unit",
                                "system_wide_telemetry_window",
                            )
                        ),
                        "model_family": model_name,
                        "test_set_evaluated": "false",
                    }
                )

                # ------------------------
                # Required parameters
                # ------------------------

                mlflow.log_params(
                    {
                        "model_name": model_name,
                        "model_seed": args.seed,
                        "git_commit": str(
                            dataset_metadata.get(
                                "source_git_commit",
                                "unknown",
                            )
                        ),
                        "dataset_sha256": str(
                            split_metadata.get(
                                "source_dataset_sha256",
                                "unknown",
                            )
                        ),
                        "split_seed": int(
                            split_metadata.get(
                                "split_seed",
                                42,
                            )
                        ),
                        "feature_count": len(
                            feature_columns
                        ),
                        "train_rows": len(
                            train_df
                        ),
                        "validation_rows": len(
                            val_df
                        ),
                        "train_runs": train_df[
                            "run_id"
                        ].nunique(),
                        "validation_runs": val_df[
                            "run_id"
                        ].nunique(),
                        "target": TARGET_COLUMN,
                        "dataset_manifest_path": str(manifest_path),
                    }
                )

                mlflow.log_params(
                    model_params_for_logging(
                        model_name,
                        model,
                    )
                )

                # ------------------------
                # Dataset lineage
                # ------------------------

                mlflow.log_input(
                    train_dataset,
                    context="training",
                )
                mlflow.log_input(
                    val_dataset,
                    context="validation",
                )

                # ------------------------
                # Train
                # ------------------------

                training_started = time.perf_counter()
                model.fit(
                    X_train,
                    y_train,
                )
                training_duration_seconds = time.perf_counter() - training_started

                y_train_pred = model.predict(X_train)
                mlflow.log_metrics({
                    "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
                    "train_macro_f1": float(f1_score(
                        y_train, y_train_pred, average="macro", zero_division=0
                    )),
                    "train_weighted_f1": float(f1_score(
                        y_train, y_train_pred, average="weighted", zero_division=0
                    )),
                })
                if model_name == "logistic_regression":
                    classifier = model.named_steps["classifier"]
                    iterations_used = int(classifier.n_iter_.max())
                    mlflow.log_param("iterations_used", iterations_used)
                    mlflow.set_tag(
                        "converged",
                        str(iterations_used < classifier.max_iter).lower(),
                    )

                # ------------------------
                # Validation inference
                # ------------------------

                y_pred = model.predict(
                    X_val
                )

                accuracy = accuracy_score(
                    y_val,
                    y_pred,
                )

                macro_f1 = f1_score(
                    y_val,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )

                weighted_f1 = f1_score(
                    y_val,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )

                mlflow.log_metrics(
                    {
                        "val_accuracy": float(
                            accuracy
                        ),
                        "val_macro_f1": float(
                            macro_f1
                        ),
                        "val_weighted_f1": float(
                            weighted_f1
                        ),
                        "training_duration_seconds": float(training_duration_seconds),
                    }
                )

                # ------------------------
                # Reports / confusion matrix
                # ------------------------

                report_json_path = (
                    model_output_dir
                    / "classification_report.json"
                )
                report_csv_path = (
                    model_output_dir
                    / "classification_report.csv"
                )
                confusion_path = (
                    model_output_dir
                    / "confusion_matrix.png"
                )

                report = save_classification_report(
                    y_true=y_val,
                    y_pred=y_pred,
                    json_path=report_json_path,
                    csv_path=report_csv_path,
                )

                log_per_class_metrics(
                    report
                )

                save_confusion_matrix(
                    y_true=y_val,
                    y_pred=y_pred,
                    output_path=confusion_path,
                    model_name=model_name,
                )

                mlflow.log_artifact(
                    str(report_json_path),
                    artifact_path="evaluation",
                )
                mlflow.log_artifact(
                    str(report_csv_path),
                    artifact_path="evaluation",
                )
                mlflow.log_artifact(
                    str(confusion_path),
                    artifact_path="evaluation",
                )

                # ------------------------
                # Feature definition artifact
                # ------------------------

                feature_file = (
                    model_output_dir
                    / "feature_columns.json"
                )

                with feature_file.open(
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        feature_columns,
                        f,
                        indent=2,
                    )

                mlflow.log_artifact(
                    str(feature_file),
                    artifact_path="dataset",
                )

                if model_name == "random_forest":
                    importance_df = pd.DataFrame({
                        "feature": feature_columns,
                        "importance": model.feature_importances_,
                    }).sort_values("importance", ascending=False)
                    importance_csv = model_output_dir / "feature_importance.csv"
                    importance_png = model_output_dir / "feature_importance.png"
                    importance_df.to_csv(importance_csv, index=False)
                    top = importance_df.head(25).sort_values("importance")
                    fig, ax = plt.subplots(figsize=(10, 9))
                    ax.barh(top["feature"], top["importance"])
                    ax.set_title("Random Forest — Top 25 Feature Importances (Dataset v2)")
                    ax.set_xlabel("Mean decrease in impurity")
                    fig.tight_layout()
                    fig.savefig(importance_png, dpi=160)
                    plt.close(fig)
                    mlflow.log_artifact(str(importance_csv), artifact_path="interpretation")
                    mlflow.log_artifact(str(importance_png), artifact_path="interpretation")

                # ------------------------
                # Log trained sklearn model
                # ------------------------

                input_example = X_val.head(
                    min(5, len(X_val))
                ).copy()

                signature = infer_signature(
                    X_train,
                    model.predict(X_train.head(20)),
                )

                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    signature=signature,
                    input_example=input_example,
                )

                comparison_rows.append(
                    {
                        "model": model_name,
                        "mlflow_run_id": run.info.run_id,
                        "val_accuracy": accuracy,
                        "val_macro_f1": macro_f1,
                        "val_weighted_f1": weighted_f1,
                        "training_duration_seconds": training_duration_seconds,
                    }
                )

        # --------------------------------------------------------------
        # Comparison table
        # --------------------------------------------------------------

        comparison_df = pd.DataFrame(
            comparison_rows
        ).sort_values(
            "val_macro_f1",
            ascending=False,
        )

        comparison_path = (
            output_dir
            / "baseline_comparison.csv"
        )

        comparison_df.to_csv(
            comparison_path,
            index=False,
        )

        print()
        print("=" * 88)
        print("RootLens RCA Baseline Training — MLflow")
        print("=" * 88)

        print(f"MLflow tracking URI:   {tracking_uri}")
        print(f"MLflow database:       {db_path}")
        print(f"Experiment:            {args.experiment_name}")
        print(f"Training rows:         {len(train_df)}")
        print(f"Validation rows:       {len(val_df)}")
        print(f"Features:              {len(feature_columns)}")
        print(f"Test set evaluated:    NO")
        print()

        print("VALIDATION RESULTS")
        print("-" * 88)

        for _, row in comparison_df.iterrows():
            print(
                f"{row['model']:<28} "
                f"accuracy={row['val_accuracy']:.4f}  "
                f"macro_f1={row['val_macro_f1']:.4f}  "
                f"weighted_f1={row['val_weighted_f1']:.4f}"
            )

        print()
        best = comparison_df.iloc[0]

        print(
            f"Best validation model: {best['model']} "
            f"(macro F1={best['val_macro_f1']:.4f})"
        )

        print()
        print(f"Comparison CSV: {comparison_path}")
        print()
        print("TEST SET REMAINS SEALED.")
        print(
            "After model selection, evaluate the selected baseline "
            "once on test.csv."
        )
        print("=" * 88)
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
