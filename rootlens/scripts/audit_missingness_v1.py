#!/usr/bin/env python3
"""
Audit missingness semantics in frozen RootLens Dataset v1.

Focus:
- latency_ms missingness vs has_requests
- error_rate missingness vs has_requests
- breakdown by service
- breakdown by run role
- breakdown by fault family
- detect suspicious missingness when has_requests == 1

Usage (from D:\RootLensAI):

    python rootlens\scripts\audit_missingness_v1.py

Outputs:
    rootlens/data/reports/dataset_v1_missingness/
        missingness_overall.csv
        missingness_by_service.csv
        missingness_by_run_role.csv
        missingness_by_fault_family.csv
        suspicious_rows.csv
        audit_summary.json

This script is READ-ONLY with respect to raw Dataset v1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Manifest is not a YAML mapping: {path}")

    return data


def accepted_run_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
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


def load_dataset_v1(
    manifest: dict[str, Any],
    repo_root: Path,
) -> pd.DataFrame:
    raw_root = Path(manifest["dataset"]["raw_data_root"])
    if not raw_root.is_absolute():
        raw_root = repo_root / raw_root
    raw_root = raw_root.resolve()

    frames: list[pd.DataFrame] = []

    for record in accepted_run_records(manifest):
        run_id = record["run_id"]
        run_dir = raw_root / run_id
        metrics_path = find_metrics_csv(run_dir)

        if metrics_path is None:
            raise FileNotFoundError(
                f"Could not identify metrics CSV for accepted run: {run_id}"
            )

        df = pd.read_csv(metrics_path)

        required = {
            "timestamp",
            "service",
            "has_requests",
            "latency_ms",
            "error_rate",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{run_id}: required audit columns missing: {sorted(missing)}"
            )

        df = df.copy()
        df["run_id"] = run_id
        df["run_role"] = record["run_role"]
        df["pair_id"] = record.get("pair_id")
        df["root_cause_service"] = record.get("root_cause_service")
        df["fault_type"] = record.get("fault_type")
        df["fault_family"] = record.get("fault_family")

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["latency_missing"] = out["latency_ms"].isna()
    out["error_rate_missing"] = out["error_rate"].isna()

    # Convert carefully in case CSV parsing ever yields bool/string values.
    has_requests = pd.to_numeric(
        out["has_requests"],
        errors="coerce",
    )

    out["has_requests_numeric"] = has_requests
    out["no_requests"] = has_requests.eq(0)
    out["has_requests_true"] = has_requests.eq(1)

    out["latency_missing_no_requests"] = (
        out["latency_missing"] & out["no_requests"]
    )
    out["latency_missing_with_requests"] = (
        out["latency_missing"] & out["has_requests_true"]
    )

    out["error_missing_no_requests"] = (
        out["error_rate_missing"] & out["no_requests"]
    )
    out["error_missing_with_requests"] = (
        out["error_rate_missing"] & out["has_requests_true"]
    )

    return out


def summarize(group: pd.DataFrame) -> dict[str, Any]:
    n = len(group)

    def pct(value: int) -> float:
        return round(100.0 * value / n, 6) if n else 0.0

    latency_missing = int(group["latency_missing"].sum())
    error_missing = int(group["error_rate_missing"].sum())

    latency_missing_no_requests = int(
        group["latency_missing_no_requests"].sum()
    )
    latency_missing_with_requests = int(
        group["latency_missing_with_requests"].sum()
    )

    error_missing_no_requests = int(
        group["error_missing_no_requests"].sum()
    )
    error_missing_with_requests = int(
        group["error_missing_with_requests"].sum()
    )

    no_requests = int(group["no_requests"].sum())
    has_requests = int(group["has_requests_true"].sum())

    return {
        "rows": n,
        "no_requests_rows": no_requests,
        "has_requests_rows": has_requests,
        "latency_missing": latency_missing,
        "latency_missing_pct": pct(latency_missing),
        "latency_missing_no_requests": latency_missing_no_requests,
        "latency_missing_with_requests": latency_missing_with_requests,
        "error_rate_missing": error_missing,
        "error_rate_missing_pct": pct(error_missing),
        "error_missing_no_requests": error_missing_no_requests,
        "error_missing_with_requests": error_missing_with_requests,
    }


def grouped_summary(
    df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for value, group in df.groupby(group_col, dropna=False):
        row = {group_col: value}
        row.update(summarize(group))
        rows.append(row)

    return pd.DataFrame(rows)


def build_suspicious_rows(df: pd.DataFrame) -> pd.DataFrame:
    suspicious = df[
        df["latency_missing_with_requests"]
        | df["error_missing_with_requests"]
    ].copy()

    keep = [
        "run_id",
        "run_role",
        "pair_id",
        "fault_family",
        "root_cause_service",
        "timestamp",
        "service",
        "has_requests",
        "request_rate",
        "latency_ms",
        "error_rate",
        "error_rps",
        "cpu",
        "memory",
        "latency_missing_with_requests",
        "error_missing_with_requests",
    ]

    existing = [c for c in keep if c in suspicious.columns]
    return suspicious[existing]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit RootLens Dataset v1 missingness semantics."
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
        default=Path("rootlens/data/reports/dataset_v1_missingness"),
        help="Directory for audit reports.",
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
        print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    try:
        manifest = load_yaml(manifest_path)
        df = load_dataset_v1(
            manifest=manifest,
            repo_root=repo_root,
        )
        df = add_missingness_flags(df)

        overall = summarize(df)
        overall_df = pd.DataFrame([overall])

        by_service = grouped_summary(df, "service")
        by_run_role = grouped_summary(df, "run_role")
        by_fault_family = grouped_summary(df, "fault_family")
        suspicious = build_suspicious_rows(df)

        output_dir.mkdir(parents=True, exist_ok=True)

        overall_df.to_csv(output_dir / "missingness_overall.csv", index=False)
        by_service.to_csv(output_dir / "missingness_by_service.csv", index=False)
        by_run_role.to_csv(output_dir / "missingness_by_run_role.csv", index=False)
        by_fault_family.to_csv(
            output_dir / "missingness_by_fault_family.csv",
            index=False,
        )
        suspicious.to_csv(output_dir / "suspicious_rows.csv", index=False)

        total_latency_missing = overall["latency_missing"]
        total_error_missing = overall["error_rate_missing"]

        latency_expected_share = (
            overall["latency_missing_no_requests"] / total_latency_missing
            if total_latency_missing
            else 1.0
        )

        error_expected_share = (
            overall["error_missing_no_requests"] / total_error_missing
            if total_error_missing
            else 1.0
        )

        summary = {
            "dataset_name": manifest["dataset"].get("name"),
            "dataset_version": manifest["dataset"].get("version"),
            "git_commit": manifest["dataset"].get("git_commit"),
            "rows_audited": len(df),
            "latency_missing": total_latency_missing,
            "latency_missing_no_requests": overall[
                "latency_missing_no_requests"
            ],
            "latency_missing_with_requests": overall[
                "latency_missing_with_requests"
            ],
            "latency_missing_explained_by_no_requests_pct": round(
                100.0 * latency_expected_share,
                6,
            ),
            "error_rate_missing": total_error_missing,
            "error_missing_no_requests": overall[
                "error_missing_no_requests"
            ],
            "error_missing_with_requests": overall[
                "error_missing_with_requests"
            ],
            "error_missing_explained_by_no_requests_pct": round(
                100.0 * error_expected_share,
                6,
            ),
            "suspicious_rows": len(suspicious),
            "recommended_interpretation": (
                "If missing latency/error_rate occurs only or overwhelmingly "
                "when has_requests == 0, treat missingness as telemetry semantics "
                "rather than collection failure. Preserve has_requests and use "
                "a deterministic imputation policy during feature construction."
            ),
        }

        with (output_dir / "audit_summary.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2)

        print()
        print("=" * 78)
        print("RootLens Dataset v1 Missingness Audit")
        print("=" * 78)
        print(f"Rows audited:                         {len(df)}")
        print()
        print("LATENCY")
        print("-" * 78)
        print(
            f"Missing latency_ms:                   "
            f"{overall['latency_missing']}"
        )
        print(
            f"Missing latency with has_requests=0: "
            f"{overall['latency_missing_no_requests']}"
        )
        print(
            f"Missing latency with has_requests=1: "
            f"{overall['latency_missing_with_requests']}"
        )
        print(
            f"Explained by no requests:            "
            f"{summary['latency_missing_explained_by_no_requests_pct']:.4f}%"
        )
        print()
        print("ERROR RATE")
        print("-" * 78)
        print(
            f"Missing error_rate:                   "
            f"{overall['error_rate_missing']}"
        )
        print(
            f"Missing error with has_requests=0:   "
            f"{overall['error_missing_no_requests']}"
        )
        print(
            f"Missing error with has_requests=1:   "
            f"{overall['error_missing_with_requests']}"
        )
        print(
            f"Explained by no requests:            "
            f"{summary['error_missing_explained_by_no_requests_pct']:.4f}%"
        )
        print()
        print("SUSPICIOUS")
        print("-" * 78)
        print(
            f"Rows with missing latency/error while has_requests=1: "
            f"{len(suspicious)}"
        )
        print()
        print(f"Reports saved to: {output_dir}")
        print("=" * 78)
        print()

        # Important: suspicious rows do not automatically fail the audit.
        # We inspect them before deciding the final imputation policy.
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())