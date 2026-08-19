#scripts/validate_dataset_manifest.py
#!/usr/bin/env python3
"""
Validate the frozen RootLens Dataset v1 manifest against raw run folders.

Usage (from D:\RootLensAI):

    python rootlens\scripts\validate_dataset_manifest.py

Optional:

    python rootlens\scripts\validate_dataset_manifest.py \
        --manifest rootlens\data\manifests\dataset_v1.yaml \
        --repo-root D:\RootLensAI

Exit codes:
    0 = validation PASS
    1 = validation FAIL
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    scope: str
    message: str


@dataclass
class ValidationResult:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    def error(self, scope: str, message: str) -> None:
        self.errors.append(ValidationIssue(scope, message))

    def warning(self, scope: str, message: str) -> None:
        self.warnings.append(ValidationIssue(scope, message))

    @property
    def passed(self) -> bool:
        return not self.errors


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
    Flatten accepted healthy runs and accepted fault/recovery pairs into
    one record per physical run folder.
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


def collect_explicitly_excluded_ids(manifest: dict[str, Any]) -> set[str]:
    excluded: set[str] = set()

    for item in manifest.get("excluded", []):
        for key in ("run_id", "fault_run_id", "recovery_run_id"):
            value = item.get(key)
            if value:
                excluded.add(str(value))

    return excluded


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def find_metrics_csv(run_dir: Path) -> Path | None:
    """
    Prefer conventional RootLens names, then fall back to the only CSV
    in the directory.

    We intentionally refuse to guess if multiple unknown CSV files exist.
    """
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


def find_metadata_json(run_dir: Path) -> Path | None:
    preferred = run_dir / "metadata.json"
    if preferred.is_file():
        return preferred

    jsons = sorted(run_dir.glob("*.json"))
    if len(jsons) == 1:
        return jsons[0]

    return None


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def resolve_column(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    """
    Resolve a logical column by exact case-insensitive match.
    """
    lookup = {str(c).strip().lower(): str(c) for c in columns}

    for candidate in candidates:
        match = lookup.get(candidate.lower())
        if match:
            return match

    return None


TIMESTAMP_CANDIDATES = (
    "timestamp",
    "time",
    "window_timestamp",
    "window_start",
    "ts",
)

SERVICE_CANDIDATES = (
    "service_name",
    "service",
    "service.name",
)


def validate_dataframe_structure(
    df: pd.DataFrame,
    run_id: str,
    expected_rows: int,
    expected_windows: int,
    expected_services: list[str],
    result: ValidationResult,
) -> None:
    scope = f"run:{run_id}"

    if len(df) != expected_rows:
        result.error(
            scope,
            f"Expected {expected_rows} rows, found {len(df)}.",
        )

    timestamp_col = resolve_column(df.columns, TIMESTAMP_CANDIDATES)
    service_col = resolve_column(df.columns, SERVICE_CANDIDATES)

    if timestamp_col is None:
        result.error(
            scope,
            "Could not find timestamp column. "
            f"Tried: {', '.join(TIMESTAMP_CANDIDATES)}. "
            f"Columns found: {list(df.columns)}",
        )

    if service_col is None:
        result.error(
            scope,
            "Could not find service column. "
            f"Tried: {', '.join(SERVICE_CANDIDATES)}. "
            f"Columns found: {list(df.columns)}",
        )

    if timestamp_col is None or service_col is None:
        return

    # Missing keys are never acceptable for structural Dataset v1 membership.
    if df[timestamp_col].isna().any():
        result.error(scope, f"Missing values found in '{timestamp_col}'.")

    if df[service_col].isna().any():
        result.error(scope, f"Missing values found in '{service_col}'.")

    # Duplicate service/window records would create leakage and count distortion.
    duplicate_mask = df.duplicated(
        subset=[timestamp_col, service_col],
        keep=False,
    )
    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count:
        result.error(
            scope,
            f"Found {duplicate_count} rows participating in duplicate "
            f"({timestamp_col}, {service_col}) keys.",
        )

    unique_windows = df[timestamp_col].nunique(dropna=False)

    if unique_windows != expected_windows:
        result.error(
            scope,
            f"Expected {expected_windows} unique windows, "
            f"found {unique_windows}.",
        )

    observed_services = set(df[service_col].astype(str).unique())
    expected_service_set = set(expected_services)

    missing_services = sorted(expected_service_set - observed_services)
    extra_services = sorted(observed_services - expected_service_set)

    if missing_services:
        result.error(
            scope,
            f"Missing frozen services: {missing_services}",
        )

    if extra_services:
        result.error(
            scope,
            f"Unexpected services not in frozen service set: {extra_services}",
        )

    # Exactly 12 rows/services per window.
    counts = df.groupby(timestamp_col, dropna=False)[service_col].nunique()
    bad_windows = counts[counts != len(expected_services)]

    if not bad_windows.empty:
        preview = ", ".join(
            f"{idx} -> {count}"
            for idx, count in bad_windows.head(10).items()
        )
        result.error(
            scope,
            f"{len(bad_windows)} windows do not contain exactly "
            f"{len(expected_services)} unique services. "
            f"First mismatches: {preview}",
        )


def validate_metadata(
    metadata_path: Path | None,
    run_id: str,
    result: ValidationResult,
) -> None:
    """
    Metadata is checked when present.

    We do not fail merely because metadata.json is absent, because some earlier
    RootLens runs may encode provenance differently. The manifest remains the
    authoritative source of Dataset-v1 membership.
    """
    if metadata_path is None:
        result.warning(
            f"run:{run_id}",
            "No unambiguous metadata JSON found; structural CSV validation "
            "was still performed.",
        )
        return

    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as exc:
        result.error(
            f"run:{run_id}",
            f"Could not read metadata file {metadata_path.name}: {exc}",
        )
        return

    if not isinstance(metadata, dict):
        result.warning(
            f"run:{run_id}",
            f"{metadata_path.name} is not a JSON object; skipping run-id check.",
        )
        return

    # Different collection versions may use different metadata keys.
    run_id_keys = (
        "run_id",
        "runId",
        "experiment_id",
        "experimentId",
    )

    metadata_run_id = None
    for key in run_id_keys:
        if key in metadata and metadata[key] is not None:
            metadata_run_id = str(metadata[key])
            break

    if metadata_run_id is not None and metadata_run_id != run_id:
        result.error(
            f"run:{run_id}",
            f"Metadata run ID '{metadata_run_id}' does not match "
            f"manifest/folder run ID '{run_id}'.",
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_manifest(
    manifest: dict[str, Any],
    repo_root: Path,
) -> tuple[ValidationResult, dict[str, Any]]:
    result = ValidationResult()

    dataset = manifest.get("dataset", {})
    protocol = manifest.get("collection_protocol", {})
    totals = manifest.get("expected_totals", {})

    raw_data_root_value = dataset.get("raw_data_root")
    if not raw_data_root_value:
        result.error("manifest", "dataset.raw_data_root is missing.")
        return result, {}

    raw_root = Path(raw_data_root_value)
    if not raw_root.is_absolute():
        raw_root = repo_root / raw_root

    raw_root = raw_root.resolve()

    if not raw_root.is_dir():
        result.error(
            "manifest",
            f"Raw data root does not exist: {raw_root}",
        )
        return result, {}

    expected_services = list(manifest.get("services", []))
    expected_rows = int(protocol.get("expected_rows_per_run", 0))
    expected_windows = int(protocol.get("expected_windows_per_run", 0))

    if not expected_services:
        result.error("manifest", "Frozen service list is empty.")

    if expected_rows <= 0:
        result.error(
            "manifest",
            "collection_protocol.expected_rows_per_run must be > 0.",
        )

    if expected_windows <= 0:
        result.error(
            "manifest",
            "collection_protocol.expected_windows_per_run must be > 0.",
        )

    records = accepted_run_records(manifest)
    run_ids = [r["run_id"] for r in records]

    # Manifest-level uniqueness.
    seen: set[str] = set()
    duplicates: set[str] = set()

    for run_id in run_ids:
        if run_id in seen:
            duplicates.add(run_id)
        seen.add(run_id)

    if duplicates:
        result.error(
            "manifest",
            f"Duplicate accepted run IDs in manifest: {sorted(duplicates)}",
        )

    explicitly_excluded = collect_explicitly_excluded_ids(manifest)

    accidental_excluded_overlap = sorted(set(run_ids) & explicitly_excluded)
    if accidental_excluded_overlap:
        result.error(
            "manifest",
            "Accepted runs also appear in the explicit exclusion/quarantine "
            f"list: {accidental_excluded_overlap}",
        )

    smoke_accepted = sorted(
        run_id for run_id in run_ids if "smoke" in run_id.lower()
    )
    if smoke_accepted:
        result.error(
            "manifest",
            f"Smoke-test runs are marked accepted: {smoke_accepted}",
        )

    role_counts = {
        "healthy": sum(r["run_role"] == "healthy" for r in records),
        "fault": sum(r["run_role"] == "fault" for r in records),
        "recovery": sum(r["run_role"] == "recovery" for r in records),
    }

    expected_healthy = int(totals.get("accepted_healthy_runs", 0))
    expected_fault = int(totals.get("accepted_fault_runs", 0))
    expected_recovery = int(totals.get("accepted_recovery_runs", 0))
    expected_total = int(totals.get("accepted_runs_total", 0))

    expected_role_counts = {
        "healthy": expected_healthy,
        "fault": expected_fault,
        "recovery": expected_recovery,
    }

    for role, observed in role_counts.items():
        expected = expected_role_counts[role]
        if observed != expected:
            result.error(
                "manifest",
                f"Expected {expected} accepted {role} runs, found {observed}.",
            )

    if len(records) != expected_total:
        result.error(
            "manifest",
            f"Expected {expected_total} total accepted runs, "
            f"found {len(records)}.",
        )

    # Physical run validation.
    observed_total_rows = 0
    missing_run_dirs: list[str] = []
    metrics_files_found = 0

    for record in records:
        run_id = record["run_id"]
        run_dir = raw_root / run_id

        if not run_dir.is_dir():
            missing_run_dirs.append(run_id)
            result.error(
                f"run:{run_id}",
                f"Accepted run directory does not exist: {run_dir}",
            )
            continue

        metrics_path = find_metrics_csv(run_dir)
        if metrics_path is None:
            result.error(
                f"run:{run_id}",
                "Could not identify a unique metrics CSV file.",
            )
            continue

        metrics_files_found += 1

        try:
            df = pd.read_csv(metrics_path)
        except Exception as exc:
            result.error(
                f"run:{run_id}",
                f"Failed to read {metrics_path.name}: {exc}",
            )
            continue

        observed_total_rows += len(df)

        validate_dataframe_structure(
            df=df,
            run_id=run_id,
            expected_rows=expected_rows,
            expected_windows=expected_windows,
            expected_services=expected_services,
            result=result,
        )

        validate_metadata(
            metadata_path=find_metadata_json(run_dir),
            run_id=run_id,
            result=result,
        )

    expected_rows_total = int(totals.get("expected_rows_total", 0))

    if not missing_run_dirs and metrics_files_found == len(records):
        if observed_total_rows != expected_rows_total:
            result.error(
                "dataset",
                f"Expected {expected_rows_total} combined rows, "
                f"found {observed_total_rows}.",
            )

    summary = {
        "dataset_name": dataset.get("name"),
        "dataset_version": dataset.get("version"),
        "dataset_status": dataset.get("status"),
        "git_commit": dataset.get("git_commit"),
        "raw_data_root": str(raw_root),
        "accepted_healthy_runs": role_counts["healthy"],
        "accepted_fault_runs": role_counts["fault"],
        "accepted_recovery_runs": role_counts["recovery"],
        "accepted_total_runs": len(records),
        "metrics_files_found": metrics_files_found,
        "expected_rows_total": expected_rows_total,
        "observed_rows_total": observed_total_rows,
        "missing_accepted_runs": len(missing_run_dirs),
        "explicitly_excluded_overlap": len(accidental_excluded_overlap),
        "smoke_runs_accepted": len(smoke_accepted),
        "errors": len(result.errors),
        "warnings": len(result.warnings),
        "status": "PASS" if result.passed else "FAIL",
    }

    return result, summary


# ---------------------------------------------------------------------------
# CLI output
# ---------------------------------------------------------------------------

def print_summary(summary: dict[str, Any]) -> None:
    print()
    print("=" * 72)
    print("RootLens Dataset Manifest Validation")
    print("=" * 72)

    print(f"Dataset:                 {summary.get('dataset_name')}")
    print(f"Version:                 {summary.get('dataset_version')}")
    print(f"Manifest status:         {summary.get('dataset_status')}")
    print(f"Git commit:              {summary.get('git_commit')}")
    print(f"Raw data root:           {summary.get('raw_data_root')}")
    print()

    print(f"Accepted healthy runs:   {summary.get('accepted_healthy_runs')}")
    print(f"Accepted fault runs:     {summary.get('accepted_fault_runs')}")
    print(f"Accepted recovery runs:  {summary.get('accepted_recovery_runs')}")
    print(f"Accepted total runs:     {summary.get('accepted_total_runs')}")
    print()

    print(f"Metrics files found:     {summary.get('metrics_files_found')}")
    print(f"Expected rows:           {summary.get('expected_rows_total')}")
    print(f"Observed rows:           {summary.get('observed_rows_total')}")
    print()

    print(f"Missing accepted runs:   {summary.get('missing_accepted_runs')}")
    print(f"Excluded overlap:        {summary.get('explicitly_excluded_overlap')}")
    print(f"Smoke runs accepted:     {summary.get('smoke_runs_accepted')}")
    print(f"Warnings:                {summary.get('warnings')}")
    print(f"Errors:                  {summary.get('errors')}")
    print()

    print(f"VALIDATION STATUS:       {summary.get('status')}")
    print("=" * 72)
    print()


def print_issues(result: ValidationResult) -> None:
    if result.errors:
        print("ERRORS")
        print("-" * 72)
        for issue in result.errors:
            print(f"[{issue.scope}] {issue.message}")
        print()

    if result.warnings:
        print("WARNINGS")
        print("-" * 72)
        for issue in result.warnings:
            print(f"[{issue.scope}] {issue.message}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate frozen RootLens Dataset v1 manifest."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("rootlens/data/manifests/dataset_v1.yaml"),
        help=(
            "Path to dataset manifest. "
            "Default: rootlens/data/manifests/dataset_v1.yaml"
        ),
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help=(
            "RootLens repository root. "
            "Default: current working directory."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = args.repo_root.resolve()

    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest_path = manifest_path.resolve()

    if not manifest_path.is_file():
        print(f"ERROR: Manifest file not found: {manifest_path}", file=sys.stderr)
        return 1

    try:
        manifest = load_yaml(manifest_path)
    except Exception as exc:
        print(
            f"ERROR: Failed to load manifest {manifest_path}: {exc}",
            file=sys.stderr,
        )
        return 1

    result, summary = validate_manifest(
        manifest=manifest,
        repo_root=repo_root,
    )

    print_summary(summary)
    print_issues(result)

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())