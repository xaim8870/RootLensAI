#!/usr/bin/env python3
"""
Inspect frozen RootLens Dataset v1 without modifying data.

Purpose:
- Read Dataset v1 membership from dataset_v1.yaml
- Inspect only accepted runs
- Summarize CSV schema consistency
- Report dtypes, missingness, numeric ranges
- Report per-column presence across runs
- Identify metadata-like / leakage-risk columns for later review
- Save machine-readable inspection artifacts

Usage (from D:\RootLensAI):

    python rootlens\scripts\inspect_dataset_v1.py

Optional:

    python rootlens\scripts\inspect_dataset_v1.py ^
        --manifest rootlens\data\manifests\dataset_v1.yaml ^
        --repo-root D:\RootLensAI

Outputs:
    rootlens/data/reports/dataset_v1_inspection/
        schema_summary.csv
        column_presence.csv
        missingness_summary.csv
        numeric_summary.csv
        categorical_summary.csv
        run_summary.csv
        inspection_summary.json

This script is READ-ONLY with respect to raw Dataset v1.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Manifest is not a YAML mapping: {path}")

    return data


def accepted_run_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flatten accepted healthy runs and fault/recovery pairs into one record
    per physical run.
    """
    records: list[dict[str, Any]] = []
    accepted = manifest.get("accepted", {})

    for item in accepted.get("healthy", []):
        records.append(
            {
                "run_id": item["run_id"],
                "run_role": "healthy",
                "condition": item.get("condition", "healthy"),
                "pair_id": None,
                "root_cause_service": "healthy",
                "fault_type": None,
                "fault_family": "healthy",
            }
        )

    for pair in accepted.get("fault_pairs", []):
        common = {
            "pair_id": pair.get("pair_id"),
            "root_cause_service": pair.get("root_cause_service"),
            "fault_type": pair.get("fault_type"),
            "fault_family": pair.get("fault_family"),
        }

        records.append(
            {
                **common,
                "run_id": pair["fault_run_id"],
                "run_role": "fault",
                "condition": pair.get("fault_condition", "fault"),
            }
        )

        records.append(
            {
                **common,
                "run_id": pair["recovery_run_id"],
                "run_role": "recovery",
                "condition": pair.get("recovery_condition", "recovery"),
            }
        )

    return records


# ---------------------------------------------------------------------------
# File discovery
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
# Heuristics
# ---------------------------------------------------------------------------

METADATA_LIKE_NAMES = {
    "run_id",
    "condition",
    "pair_id",
    "fault_service",
    "fault_type",
    "fault_family",
    "root_cause_service",
    "label",
    "target",
    "class",
    "experiment_id",
    "experiment",
}

TIMESTAMP_LIKE_NAMES = {
    "timestamp",
    "time",
    "ts",
    "window_timestamp",
    "window_start",
    "window_end",
}

SERVICE_LIKE_NAMES = {
    "service",
    "service_name",
    "service.name",
}


def normalize_name(name: str) -> str:
    return str(name).strip().lower()


def classify_column_for_review(name: str) -> str:
    """
    Conservative heuristic only.
    This does NOT decide the final feature set.
    """
    n = normalize_name(name)

    if n in METADATA_LIKE_NAMES:
        return "metadata_or_label_risk"

    if n in TIMESTAMP_LIKE_NAMES:
        return "time_key_review"

    if n in SERVICE_LIKE_NAMES:
        return "entity_key_review"

    # Some common provenance patterns.
    if any(token in n for token in ("run_id", "fault_", "condition", "label", "target")):
        return "metadata_or_label_risk"

    return "candidate_feature_review"


def series_semantic_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical_or_text"


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def inspect_dataset(
    manifest: dict[str, Any],
    repo_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    dataset = manifest.get("dataset", {})
    raw_root_value = dataset.get("raw_data_root")

    if not raw_root_value:
        raise ValueError("dataset.raw_data_root is missing from manifest.")

    raw_root = Path(raw_root_value)
    if not raw_root.is_absolute():
        raw_root = repo_root / raw_root
    raw_root = raw_root.resolve()

    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw data root does not exist: {raw_root}")

    records = accepted_run_records(manifest)

    run_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    presence_counter: Counter[str] = Counter()
    dtype_counter: dict[str, Counter[str]] = defaultdict(Counter)

    missing_accumulator: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "rows_present": 0,
            "missing_count": 0,
            "non_missing_count": 0,
        }
    )

    numeric_values: dict[str, list[pd.Series]] = defaultdict(list)
    categorical_values: dict[str, Counter[str]] = defaultdict(Counter)

    canonical_schema_signatures: Counter[tuple[str, ...]] = Counter()

    total_rows = 0
    loaded_runs = 0

    for record in records:
        run_id = record["run_id"]
        run_dir = raw_root / run_id

        metrics_path = find_metrics_csv(run_dir)
        if metrics_path is None:
            raise FileNotFoundError(
                f"Could not identify metrics CSV for accepted run: {run_id}"
            )

        df = pd.read_csv(metrics_path)
        loaded_runs += 1
        total_rows += len(df)

        columns = [str(c) for c in df.columns]
        schema_signature = tuple(columns)
        canonical_schema_signatures[schema_signature] += 1

        run_rows.append(
            {
                "run_id": run_id,
                "run_role": record["run_role"],
                "condition": record.get("condition"),
                "pair_id": record.get("pair_id"),
                "root_cause_service": record.get("root_cause_service"),
                "fault_type": record.get("fault_type"),
                "fault_family": record.get("fault_family"),
                "rows": len(df),
                "columns": len(columns),
                "metrics_file": str(metrics_path),
            }
        )

        for col in columns:
            presence_counter[col] += 1
            dtype_name = str(df[col].dtype)
            dtype_counter[col][dtype_name] += 1

            sem_type = series_semantic_type(df[col])

            schema_rows.append(
                {
                    "run_id": run_id,
                    "column": col,
                    "pandas_dtype": dtype_name,
                    "semantic_type": sem_type,
                    "review_class": classify_column_for_review(col),
                }
            )

            missing_count = int(df[col].isna().sum())
            non_missing_count = int(df[col].notna().sum())

            missing_accumulator[col]["rows_present"] += len(df)
            missing_accumulator[col]["missing_count"] += missing_count
            missing_accumulator[col]["non_missing_count"] += non_missing_count

            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_values[col].append(pd.to_numeric(df[col], errors="coerce"))
            else:
                # Cap collection indirectly via Counter update; still enough for top values.
                vals = df[col].dropna().astype(str)
                categorical_values[col].update(vals.tolist())

    all_columns = sorted(presence_counter)

    # ------------------------------------------------------------------
    # Column presence
    # ------------------------------------------------------------------

    presence_rows = []
    for col in all_columns:
        dtype_counts = dtype_counter[col]
        presence_rows.append(
            {
                "column": col,
                "runs_present": presence_counter[col],
                "runs_total": loaded_runs,
                "presence_pct": round(
                    100.0 * presence_counter[col] / loaded_runs, 4
                ) if loaded_runs else 0.0,
                "dtype_variants": "; ".join(
                    f"{dtype}:{count}"
                    for dtype, count in sorted(dtype_counts.items())
                ),
                "review_class": classify_column_for_review(col),
            }
        )

    column_presence_df = pd.DataFrame(presence_rows)

    # ------------------------------------------------------------------
    # Missingness
    # ------------------------------------------------------------------

    missing_rows = []
    for col in all_columns:
        acc = missing_accumulator[col]
        rows_present = int(acc["rows_present"])
        missing_count = int(acc["missing_count"])
        non_missing_count = int(acc["non_missing_count"])

        missing_rows.append(
            {
                "column": col,
                "rows_where_column_present": rows_present,
                "missing_count": missing_count,
                "non_missing_count": non_missing_count,
                "missing_pct": round(
                    100.0 * missing_count / rows_present, 6
                ) if rows_present else None,
                "review_class": classify_column_for_review(col),
            }
        )

    missingness_df = pd.DataFrame(missing_rows)

    # ------------------------------------------------------------------
    # Numeric summary
    # ------------------------------------------------------------------

    numeric_rows = []

    for col, pieces in sorted(numeric_values.items()):
        combined = pd.concat(pieces, ignore_index=True)
        clean = combined.dropna()

        if clean.empty:
            numeric_rows.append(
                {
                    "column": col,
                    "count": 0,
                    "missing_count": int(combined.isna().sum()),
                    "min": None,
                    "p01": None,
                    "p05": None,
                    "median": None,
                    "mean": None,
                    "p95": None,
                    "p99": None,
                    "max": None,
                    "std": None,
                    "zero_count": 0,
                    "negative_count": 0,
                    "review_class": classify_column_for_review(col),
                }
            )
            continue

        q = clean.quantile([0.01, 0.05, 0.5, 0.95, 0.99])

        numeric_rows.append(
            {
                "column": col,
                "count": int(clean.count()),
                "missing_count": int(combined.isna().sum()),
                "min": float(clean.min()),
                "p01": float(q.loc[0.01]),
                "p05": float(q.loc[0.05]),
                "median": float(q.loc[0.5]),
                "mean": float(clean.mean()),
                "p95": float(q.loc[0.95]),
                "p99": float(q.loc[0.99]),
                "max": float(clean.max()),
                "std": float(clean.std()) if len(clean) > 1 else 0.0,
                "zero_count": int((clean == 0).sum()),
                "negative_count": int((clean < 0).sum()),
                "review_class": classify_column_for_review(col),
            }
        )

    numeric_summary_df = pd.DataFrame(numeric_rows)

    # ------------------------------------------------------------------
    # Categorical/text summary
    # ------------------------------------------------------------------

    categorical_rows = []

    for col, counter in sorted(categorical_values.items()):
        total_non_missing = sum(counter.values())
        unique_count = len(counter)
        top = counter.most_common(10)

        categorical_rows.append(
            {
                "column": col,
                "non_missing_count": total_non_missing,
                "unique_count": unique_count,
                "top_values": json.dumps(top, ensure_ascii=False),
                "review_class": classify_column_for_review(col),
            }
        )

    categorical_summary_df = pd.DataFrame(categorical_rows)

    run_summary_df = pd.DataFrame(run_rows)
    schema_summary_df = pd.DataFrame(schema_rows)

    unique_schema_count = len(canonical_schema_signatures)
    most_common_schema = None
    most_common_schema_runs = 0

    if canonical_schema_signatures:
        most_common_schema, most_common_schema_runs = canonical_schema_signatures.most_common(1)[0]

    candidate_feature_columns = [
        col for col in all_columns
        if classify_column_for_review(col) == "candidate_feature_review"
    ]

    risky_columns = [
        col for col in all_columns
        if classify_column_for_review(col) != "candidate_feature_review"
    ]

    summary = {
        "dataset_name": dataset.get("name"),
        "dataset_version": dataset.get("version"),
        "git_commit": dataset.get("git_commit"),
        "raw_data_root": str(raw_root),
        "accepted_runs_expected": len(records),
        "accepted_runs_loaded": loaded_runs,
        "total_rows_loaded": total_rows,
        "unique_columns": len(all_columns),
        "all_columns": all_columns,
        "unique_schema_count": unique_schema_count,
        "most_common_schema_runs": most_common_schema_runs,
        "most_common_schema": list(most_common_schema) if most_common_schema else [],
        "candidate_feature_columns_for_review": candidate_feature_columns,
        "metadata_or_key_columns_for_review": risky_columns,
        "note": (
            "Column classifications are heuristics for review only. "
            "No final ML feature set is selected by this script."
        ),
    }

    tables = {
        "run_summary": run_summary_df,
        "schema_summary": schema_summary_df,
        "column_presence": column_presence_df,
        "missingness_summary": missingness_df,
        "numeric_summary": numeric_summary_df,
        "categorical_summary": categorical_summary_df,
    }

    return tables, summary


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_outputs(
    tables: dict[str, pd.DataFrame],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in tables.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)

    with (output_dir / "inspection_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def print_console_report(
    tables: dict[str, pd.DataFrame],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    print()
    print("=" * 78)
    print("RootLens Dataset v1 Inspection")
    print("=" * 78)
    print(f"Dataset:                  {summary['dataset_name']}")
    print(f"Version:                  {summary['dataset_version']}")
    print(f"Git commit:               {summary['git_commit']}")
    print(f"Accepted runs loaded:     {summary['accepted_runs_loaded']}")
    print(f"Total rows loaded:        {summary['total_rows_loaded']}")
    print(f"Unique columns:           {summary['unique_columns']}")
    print(f"Unique schema signatures: {summary['unique_schema_count']}")
    print(f"Output directory:         {output_dir}")
    print()

    print("COLUMNS")
    print("-" * 78)
    for col in summary["all_columns"]:
        review = classify_column_for_review(col)
        print(f"{col:<40} {review}")
    print()

    presence_df = tables["column_presence"]
    inconsistent = presence_df[
        presence_df["runs_present"] != presence_df["runs_total"]
    ]

    print("SCHEMA CONSISTENCY")
    print("-" * 78)
    if inconsistent.empty:
        print("All discovered columns are present in all accepted runs.")
    else:
        print("Columns not present in every accepted run:")
        for _, row in inconsistent.iterrows():
            print(
                f"  {row['column']}: "
                f"{int(row['runs_present'])}/{int(row['runs_total'])} runs"
            )
    print()

    missing_df = tables["missingness_summary"].sort_values(
        "missing_pct", ascending=False
    )

    print("TOP MISSINGNESS")
    print("-" * 78)
    if missing_df.empty:
        print("No columns found.")
    else:
        for _, row in missing_df.head(15).iterrows():
            print(
                f"{row['column']:<40} "
                f"{row['missing_pct']:>10.4f}%"
            )
    print()

    numeric_df = tables["numeric_summary"]
    print("NUMERIC COLUMNS")
    print("-" * 78)
    if numeric_df.empty:
        print("No numeric columns found.")
    else:
        for _, row in numeric_df.iterrows():
            print(
                f"{row['column']:<30} "
                f"min={row['min']!s:<14} "
                f"median={row['median']!s:<14} "
                f"max={row['max']!s}"
            )
    print()

    print("IMPORTANT")
    print("-" * 78)
    print(
        "This script did NOT choose the final feature set, create labels, "
        "split data, scale features, or train a model."
    )
    print("=" * 78)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect frozen RootLens Dataset v1."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("rootlens/data/manifests/dataset_v1.yaml"),
        help="Path to Dataset v1 manifest.",
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
        default=Path("rootlens/data/reports/dataset_v1_inspection"),
        help="Directory for inspection reports.",
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
        tables, summary = inspect_dataset(
            manifest=manifest,
            repo_root=repo_root,
        )
        save_outputs(
            tables=tables,
            summary=summary,
            output_dir=output_dir,
        )
        print_console_report(
            tables=tables,
            summary=summary,
            output_dir=output_dir,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())