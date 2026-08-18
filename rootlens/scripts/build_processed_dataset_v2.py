#!/usr/bin/env python3
"""
Build the frozen processed RootLens Dataset v2 representation.

Input:
    rootlens/data/manifests/dataset_v2.yaml
    rootlens/data/raw/<accepted run folders>/

Output:
    rootlens/data/processed/dataset_v2/
        processed_dataset_v2.csv
        processed_dataset_v2_metadata.json
        feature_columns.txt

Primary ML task:
    Root-cause service classification

Sample unit:
    One system-wide telemetry window

Training-eligible run roles:
    healthy + fault

Excluded from initial classifier dataset:
    recovery runs

Feature construction:
    12 services × 7 telemetry metrics = 84 features

Frozen telemetry metrics:
    cpu
    memory
    request_rate
    has_requests
    latency_ms
    error_rps
    error_rate

Frozen missingness policy:
    latency_ms NaN -> 0
    error_rate NaN -> 0
    has_requests is retained as a feature

IMPORTANT:
    This script does NOT split the dataset and does NOT train a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


FEATURE_METRICS = [
    "cpu",
    "memory",
    "request_rate",
    "has_requests",
    "latency_ms",
    "error_rps",
    "error_rate",
]

EXPECTED_SERVICES = [
    "frontend",
    "frontend-proxy",
    "checkout",
    "payment",
    "cart",
    "currency",
    "shipping",
    "product-catalog",
    "recommendation",
    "email",
    "ad",
    "quote",
]

EXPECTED_TRAINING_RUNS = 40
EXPECTED_WINDOWS_PER_RUN = 60
EXPECTED_SAMPLES = EXPECTED_TRAINING_RUNS * EXPECTED_WINDOWS_PER_RUN
EXPECTED_FEATURES = len(EXPECTED_SERVICES) * len(FEATURE_METRICS)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Manifest is not a YAML mapping: {path}")

    return data


def training_run_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Return only runs eligible for the initial RCA classifier:
      - healthy
      - active fault
    Recovery runs are deliberately excluded.
    """
    records: list[dict[str, Any]] = []
    accepted = manifest.get("accepted", {})

    # Healthy runs
    for item in accepted.get("healthy", []):
        records.append(
            {
                "run_id": item["run_id"],
                "run_role": "healthy",
                "condition": item.get("condition", "healthy"),
                "pair_id": None,
                "root_cause_service": "healthy",
                "fault_type": "none",
                "fault_family": "healthy",
                "is_fault": 0,
            }
        )

    # Active-fault runs only
    for pair in accepted.get("fault_pairs", []):
        records.append(
            {
                "run_id": pair["fault_run_id"],
                "run_role": "fault",
                "condition": pair.get("fault_condition", "fault"),
                "pair_id": pair.get("pair_id"),
                "root_cause_service": pair.get("root_cause_service"),
                "fault_type": pair.get("fault_type"),
                "fault_family": pair.get("fault_family"),
                "is_fault": 1,
            }
        )

    return records


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def find_metrics_csv(run_dir: Path) -> Path | None:
    preferred = (
        "metrics.csv",
        "metrics(1).csv",
        "data.csv",
        "telemetry.csv",
    )

    for name in preferred:
        path = run_dir / name
        if path.is_file():
            return path

    csvs = sorted(run_dir.glob("*.csv"))

    if len(csvs) == 1:
        return csvs[0]

    return None


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def validate_raw_run(
    df: pd.DataFrame,
    run_id: str,
) -> None:
    required = {"timestamp", "service", *FEATURE_METRICS}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{run_id}: missing required columns: {sorted(missing)}"
        )

    if len(df) != EXPECTED_WINDOWS_PER_RUN * len(EXPECTED_SERVICES):
        raise ValueError(
            f"{run_id}: expected "
            f"{EXPECTED_WINDOWS_PER_RUN * len(EXPECTED_SERVICES)} rows, "
            f"found {len(df)}"
        )

    observed_services = set(df["service"].astype(str).unique())
    expected_services = set(EXPECTED_SERVICES)

    if observed_services != expected_services:
        raise ValueError(
            f"{run_id}: service set mismatch. "
            f"Missing={sorted(expected_services - observed_services)}, "
            f"Extra={sorted(observed_services - expected_services)}"
        )

    n_windows = df["timestamp"].nunique()

    if n_windows != EXPECTED_WINDOWS_PER_RUN:
        raise ValueError(
            f"{run_id}: expected {EXPECTED_WINDOWS_PER_RUN} windows, "
            f"found {n_windows}"
        )

    duplicate_count = int(
        df.duplicated(subset=["timestamp", "service"]).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"{run_id}: found {duplicate_count} duplicate "
            "(timestamp, service) rows"
        )


def apply_missingness_policy(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Frozen baseline policy:
      latency_ms NaN -> 0
      error_rate NaN -> 0

    We preserve has_requests so the model can distinguish
    'no measurement because no traffic' from a real zero-like value.
    """
    out = df.copy()

    stats = {
        "latency_ms_missing_before": int(out["latency_ms"].isna().sum()),
        "error_rate_missing_before": int(out["error_rate"].isna().sum()),
    }

    out["latency_ms"] = out["latency_ms"].fillna(0.0)
    out["error_rate"] = out["error_rate"].fillna(0.0)

    stats["latency_ms_missing_after"] = int(out["latency_ms"].isna().sum())
    stats["error_rate_missing_after"] = int(out["error_rate"].isna().sum())

    if stats["latency_ms_missing_after"] != 0:
        raise ValueError("latency_ms still contains NaN after imputation")

    if stats["error_rate_missing_after"] != 0:
        raise ValueError("error_rate still contains NaN after imputation")

    return out, stats


def build_feature_rowset(
    df: pd.DataFrame,
    run_record: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Pivot one run from:
        60 timestamps × 12 service rows
    into:
        60 timestamps × 84 service-metric features
    """
    run_id = run_record["run_id"]

    long_df = df[
        ["timestamp", "service", *FEATURE_METRICS]
    ].copy()

    # Pivot to MultiIndex columns: (metric, service)
    pivoted = long_df.pivot(
        index="timestamp",
        columns="service",
        values=FEATURE_METRICS,
    )

    # Reindex explicitly so feature order is deterministic.
    desired_multi_columns = pd.MultiIndex.from_product(
        [FEATURE_METRICS, EXPECTED_SERVICES],
        names=["metric", "service"],
    )

    pivoted = pivoted.reindex(columns=desired_multi_columns)

    # Verify no missing feature cells remain after pivot.
    if pivoted.isna().any().any():
        missing_cells = int(pivoted.isna().sum().sum())
        raise ValueError(
            f"{run_id}: pivot produced {missing_cells} missing feature cells"
        )

    # Flatten to service__metric naming.
    #
    # Example:
    #   payment__latency_ms
    #   cart__error_rate
    #
    # We use service first because it is easier to interpret during RCA.
    flattened_columns = [
        f"{service}__{metric}"
        for metric, service in pivoted.columns
    ]

    pivoted.columns = flattened_columns
    pivoted = pivoted.reset_index()

    # Add provenance/labels.
    pivoted.insert(0, "run_id", run_id)
    pivoted.insert(1, "run_role", run_record["run_role"])
    pivoted.insert(2, "pair_id", run_record["pair_id"])
    pivoted.insert(3, "condition", run_record["condition"])
    pivoted.insert(
        4,
        "root_cause_service",
        run_record["root_cause_service"],
    )
    pivoted.insert(5, "fault_type", run_record["fault_type"])
    pivoted.insert(6, "fault_family", run_record["fault_family"])
    pivoted.insert(7, "is_fault", run_record["is_fault"])

    if len(pivoted) != EXPECTED_WINDOWS_PER_RUN:
        raise ValueError(
            f"{run_id}: expected {EXPECTED_WINDOWS_PER_RUN} processed rows, "
            f"found {len(pivoted)}"
        )

    return pivoted, flattened_columns


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_processed_dataset(
    manifest: dict[str, Any],
    repo_root: Path,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    raw_root = Path(manifest["dataset"]["raw_data_root"])

    if not raw_root.is_absolute():
        raw_root = repo_root / raw_root

    raw_root = raw_root.resolve()

    if not raw_root.is_dir():
        raise FileNotFoundError(
            f"Raw data root does not exist: {raw_root}"
        )

    records = training_run_records(manifest)

    if len(records) != EXPECTED_TRAINING_RUNS:
        raise ValueError(
            f"Expected {EXPECTED_TRAINING_RUNS} healthy+fault training runs, "
            f"found {len(records)}"
        )

    processed_frames: list[pd.DataFrame] = []
    feature_columns: list[str] | None = None

    missingness_totals = {
        "latency_ms_missing_before": 0,
        "error_rate_missing_before": 0,
        "latency_ms_missing_after": 0,
        "error_rate_missing_after": 0,
    }

    per_run_rows: dict[str, int] = {}

    for record in records:
        run_id = record["run_id"]
        run_dir = raw_root / run_id

        metrics_path = find_metrics_csv(run_dir)

        if metrics_path is None:
            raise FileNotFoundError(
                f"{run_id}: could not identify metrics CSV"
            )

        df = pd.read_csv(metrics_path)

        validate_raw_run(df, run_id)

        df, stats = apply_missingness_policy(df)

        for key, value in stats.items():
            missingness_totals[key] += value

        processed_run, run_feature_columns = build_feature_rowset(
            df=df,
            run_record=record,
        )

        if feature_columns is None:
            feature_columns = run_feature_columns
        elif run_feature_columns != feature_columns:
            raise ValueError(
                f"{run_id}: processed feature order differs from prior runs"
            )

        per_run_rows[run_id] = len(processed_run)
        processed_frames.append(processed_run)

    if feature_columns is None:
        raise ValueError("No feature columns were generated")

    processed = pd.concat(processed_frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Final invariants
    # ------------------------------------------------------------------

    if len(processed) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} processed samples, "
            f"found {len(processed)}"
        )

    if len(feature_columns) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} feature columns, "
            f"found {len(feature_columns)}"
        )

    if processed[feature_columns].isna().any().any():
        raise ValueError(
            "Processed feature matrix contains NaN values"
        )

    if processed["run_id"].nunique() != EXPECTED_TRAINING_RUNS:
        raise ValueError(
            f"Expected {EXPECTED_TRAINING_RUNS} unique training runs, "
            f"found {processed['run_id'].nunique()}"
        )

    expected_labels = {
        "healthy",
        "payment",
        "cart",
        "checkout",
        "product_catalog",
    }

    observed_labels = set(
        processed["root_cause_service"].astype(str).unique()
    )

    if observed_labels != expected_labels:
        raise ValueError(
            "Root-cause label set mismatch. "
            f"Expected={sorted(expected_labels)}, "
            f"Observed={sorted(observed_labels)}"
        )

    # Expected class counts from the frozen experiment design.
    expected_class_counts = {
        "healthy": 300,
        "payment": 600,
        "cart": 300,
        "checkout": 600,
        "product_catalog": 600,
    }

    observed_class_counts = (
        processed["root_cause_service"]
        .value_counts()
        .to_dict()
    )

    if observed_class_counts != expected_class_counts:
        raise ValueError(
            "Unexpected class distribution. "
            f"Expected={expected_class_counts}, "
            f"Observed={observed_class_counts}"
        )

    # Each run must contribute exactly 60 windows.
    run_counts = processed["run_id"].value_counts()

    bad_run_counts = run_counts[
        run_counts != EXPECTED_WINDOWS_PER_RUN
    ]

    if not bad_run_counts.empty:
        raise ValueError(
            "Some runs do not contribute exactly "
            f"{EXPECTED_WINDOWS_PER_RUN} processed windows: "
            f"{bad_run_counts.to_dict()}"
        )

    metadata = {
        "dataset_name": manifest["dataset"].get("name"),
        "dataset_version": manifest["dataset"].get("version"),
        "source_git_commit": manifest["dataset"].get("git_commit"),
        "manifest_status": manifest["dataset"].get("status"),
        "sample_unit": "system_wide_telemetry_window",
        "primary_prediction_target": "root_cause_service",
        "secondary_prediction_target": "fault_family",
        "training_run_roles": ["healthy", "fault"],
        "recovery_runs_included": False,
        "grouping_key_for_future_split": "run_id",
        "random_row_split_allowed": False,
        "services": EXPECTED_SERVICES,
        "telemetry_metrics": FEATURE_METRICS,
        "feature_count": len(feature_columns),
        "sample_count": len(processed),
        "training_run_count": processed["run_id"].nunique(),
        "windows_per_run": EXPECTED_WINDOWS_PER_RUN,
        "class_counts": observed_class_counts,
        "missingness_policy": {
            "latency_ms": "fill_nan_with_0",
            "error_rate": "fill_nan_with_0",
            "reason": (
                "Dataset v1 frozen audit showed 98.55% of missing latency_ms and "
                "100% of missing error_rate occurred when has_requests=0. "
                "has_requests is retained as a feature."
            ),
            "audit_counts": missingness_totals,
        },
        "feature_naming": "<service>__<metric>",
        "feature_columns": feature_columns,
        "notes": [
            "Recovery runs are reserved for later intervention/recovery fidelity evaluation.",
            "No train/validation/test split is created by this builder.",
            "No feature scaling is applied by this builder.",
            "No model training is performed by this builder.",
        ],
    }

    return processed, feature_columns, metadata


# ---------------------------------------------------------------------------
# CLI / outputs
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen processed RootLens Dataset v2."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("rootlens/data/manifests/dataset_v2.yaml"),
        help="Path to Dataset v2 manifest.",
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Default: current working directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rootlens/data/processed/dataset_v2"),
        help="Output directory for processed Dataset v2.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = args.repo_root.resolve()

    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest_path = manifest_path.resolve()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir = output_dir.resolve()

    if not manifest_path.is_file():
        print(
            f"ERROR: Manifest not found: {manifest_path}",
            file=sys.stderr,
        )
        return 1

    try:
        manifest = load_yaml(manifest_path)

        processed, feature_columns, metadata = build_processed_dataset(
            manifest=manifest,
            repo_root=repo_root,
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "processed_dataset_v2.csv"
        metadata_path = output_dir / "processed_dataset_v2_metadata.json"
        feature_path = output_dir / "feature_columns.txt"

        processed.to_csv(csv_path, index=False)

        # Add artifact checksum after CSV is written.
        metadata["processed_csv_sha256"] = sha256_file(csv_path)

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(
                metadata,
                f,
                indent=2,
                ensure_ascii=False,
            )

        feature_path.write_text(
            "\n".join(feature_columns) + "\n",
            encoding="utf-8",
        )

        print()
        print("=" * 78)
        print("RootLens Processed Dataset v2 Build")
        print("=" * 78)
        print(f"Training runs:           {processed['run_id'].nunique()}")
        print(f"Processed samples:       {len(processed)}")
        print(f"Telemetry features:      {len(feature_columns)}")
        print(f"Sample unit:             system-wide telemetry window")
        print(f"Primary target:          root_cause_service")
        print(f"Recovery runs included:  NO")
        print()
        print("CLASS DISTRIBUTION")
        print("-" * 78)

        class_counts = (
            processed["root_cause_service"]
            .value_counts()
            .sort_index()
        )

        for label, count in class_counts.items():
            print(f"{label:<24} {count}")

        print()
        print("MISSINGNESS POLICY")
        print("-" * 78)

        audit = metadata["missingness_policy"]["audit_counts"]

        print(
            f"latency_ms NaN before:   "
            f"{audit['latency_ms_missing_before']}"
        )
        print(
            f"latency_ms NaN after:    "
            f"{audit['latency_ms_missing_after']}"
        )
        print(
            f"error_rate NaN before:   "
            f"{audit['error_rate_missing_before']}"
        )
        print(
            f"error_rate NaN after:    "
            f"{audit['error_rate_missing_after']}"
        )

        print()
        print("OUTPUTS")
        print("-" * 78)
        print(f"CSV:       {csv_path}")
        print(f"Metadata:  {metadata_path}")
        print(f"Features:  {feature_path}")
        print()
        print(f"CSV SHA256: {metadata['processed_csv_sha256']}")
        print()
        print("BUILD STATUS: PASS")
        print("=" * 78)
        print()

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
