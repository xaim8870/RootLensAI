"""Validate RootLens metric runs without altering their collected data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_COLUMNS = {
    "timestamp", "service", "request_rate", "error_rate", "latency_ms",
    "cpu", "memory", "error_rps", "has_requests",
}
NONNEGATIVE_COLUMNS = ("request_rate", "latency_ms", "cpu", "memory", "error_rps")


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": "PASS" if self.passed else "FAIL", "detail": self.detail}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def _median(df: pd.DataFrame, service: str, column: str) -> float | None:
    values = pd.to_numeric(df.loc[df["service"] == service, column], errors="coerce").dropna()
    return None if values.empty else float(values.median())


def validate_run(
    run_dir: Path,
    config: dict[str, Any],
    condition: str,
    comparison_run_dir: Path | None = None,
) -> dict[str, Any]:
    gates: list[Gate] = []
    csv_path, metadata_path = run_dir / "metrics.csv", run_dir / "metadata.json"
    if not csv_path.is_file() or not metadata_path.is_file():
        missing = [p.name for p in (csv_path, metadata_path) if not p.is_file()]
        gates.append(Gate("run_files", False, f"missing: {', '.join(missing)}"))
        return _result(run_dir, condition, gates)

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        df = pd.read_csv(csv_path)
    except Exception as exc:
        gates.append(Gate("readable_dataset", False, str(exc)))
        return _result(run_dir, condition, gates)

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    gates.append(Gate("required_columns", not missing_columns,
                      "all required columns present" if not missing_columns else f"missing: {missing_columns}"))
    if missing_columns:
        return _result(run_dir, condition, gates)

    expected_services = list(config["expected_services"])
    collection, limits = config["collection"], config["validation"]
    expected_windows = math.ceil(collection["duration_seconds"] / collection["sampling_interval_seconds"])
    minimum_windows = math.ceil(expected_windows * limits["minimum_window_fraction"])
    window_counts = df.groupby("timestamp")["service"].nunique()
    actual_windows = int(df["timestamp"].nunique())
    expected_rows = expected_windows * len(expected_services)
    minimum_rows = minimum_windows * len(expected_services)

    gates.append(Gate("expected_windows", actual_windows >= minimum_windows,
                      f"actual={actual_windows}, minimum={minimum_windows}, configured={expected_windows}"))
    gates.append(Gate("expected_rows", len(df) >= minimum_rows,
                      f"actual={len(df)}, minimum={minimum_rows}, configured={expected_rows}"))
    wrong_windows = int((window_counts != len(expected_services)).sum())
    gates.append(Gate("services_per_window", wrong_windows == 0,
                      f"windows_with_wrong_service_count={wrong_windows}, expected_per_window={len(expected_services)}"))
    duplicate_rows = int(df.duplicated(subset=["timestamp", "service"]).sum())
    gates.append(Gate("unique_service_rows", duplicate_rows == 0,
                      f"duplicate_timestamp_service_rows={duplicate_rows}"))
    observed = set(df["service"].dropna().astype(str))
    missing_services = sorted(set(expected_services) - observed)
    unexpected_services = sorted(observed - set(expected_services))
    gates.append(Gate("expected_services", not missing_services and not unexpected_services,
                      f"missing={missing_services}, unexpected={unexpected_services}"))

    for column, limit_key in (
        ("request_rate", "maximum_missing_request_rate_fraction"),
        ("cpu", "maximum_missing_cpu_fraction"),
        ("memory", "maximum_missing_memory_fraction"),
    ):
        fraction = float(df[column].isna().mean()) if len(df) else 1.0
        limit = float(limits[limit_key])
        gates.append(Gate(f"missing_{column}", fraction <= limit, f"fraction={fraction:.4f}, maximum={limit:.4f}"))

    target = config["fault_service"]
    target_rows = df[df["service"] == target]
    active_fraction = float((target_rows["has_requests"] == 1).mean()) if len(target_rows) else 0.0
    gates.append(Gate("target_service_activity", active_fraction >= limits["minimum_target_active_fraction"],
                      f"service={target}, active_fraction={active_fraction:.4f}, minimum={limits['minimum_target_active_fraction']:.4f}"))

    negative_counts = {}
    for column in NONNEGATIVE_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")
        count = int((values < 0).sum())
        if count:
            negative_counts[column] = count
    gates.append(Gate("nonnegative_metrics", not negative_counts, f"negative_counts={negative_counts}"))
    errors = pd.to_numeric(df["error_rate"], errors="coerce")
    invalid_errors = int(((errors < 0) | (errors > limits["maximum_error_rate"])).sum())
    gates.append(Gate("error_rate_bounds", invalid_errors == 0,
                      f"invalid_rows={invalid_errors}, bounds=[0,{limits['maximum_error_rate']}]"))

    target_latency = _median(df, target, "latency_ms")
    if condition == "fault":
        threshold = float(limits["minimum_fault_target_latency_ms"])
        gates.append(Gate("intervention_visible", target_latency is not None and target_latency >= threshold,
                          f"target_median_latency_ms={target_latency}, minimum={threshold}"))
        if comparison_run_dir:
            comparison = pd.read_csv(comparison_run_dir / "metrics.csv")
            baseline = _median(comparison, target, "latency_ms")
            increase = None if target_latency is None or baseline is None else target_latency - baseline
            minimum = float(limits["minimum_latency_increase_ms"])
            gates.append(Gate("latency_increase", increase is not None and increase >= minimum,
                              f"increase_ms={increase}, minimum={minimum}, baseline_ms={baseline}"))
    elif condition == "recovery":
        if comparison_run_dir is None:
            gates.append(Gate("recovery_comparison", False, "fault comparison run is required"))
        else:
            fault_df = pd.read_csv(comparison_run_dir / "metrics.csv")
            fault_latency = _median(fault_df, target, "latency_ms")
            ratio = None if target_latency is None or not fault_latency else target_latency / fault_latency
            maximum = float(limits["maximum_recovery_to_fault_latency_ratio"])
            gates.append(Gate("recovery_direction", ratio is not None and ratio <= maximum,
                              f"recovery_to_fault_latency_ratio={ratio}, maximum={maximum}, recovery_ms={target_latency}, fault_ms={fault_latency}"))
    else:
        gates.append(Gate("condition", False, f"unsupported condition: {condition}"))

    rows_meta, windows_meta = metadata.get("rows_collected"), metadata.get("windows_collected")
    matches = rows_meta == len(df) and windows_meta == actual_windows
    gates.append(Gate("metadata_consistency", matches,
                      f"metadata_rows={rows_meta}, csv_rows={len(df)}, metadata_windows={windows_meta}, csv_windows={actual_windows}"))
    return _result(run_dir, condition, gates)


def _result(run_dir: Path, condition: str, gates: list[Gate]) -> dict[str, Any]:
    passed = bool(gates) and all(gate.passed for gate in gates)
    return {
        "schema_version": 1,
        "run_id": run_dir.name,
        "condition": condition,
        "status": "PASS" if passed else "FAIL",
        "research_valid": passed,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "gates": [gate.as_dict() for gate in gates],
    }


def write_result(run_dir: Path, result: dict[str, Any]) -> Path:
    output = run_dir / "validation.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing validation result: {output}")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--condition", required=True, choices=("fault", "recovery"))
    parser.add_argument("--comparison-run-dir", type=Path)
    parser.add_argument("--write-result", action="store_true")
    args = parser.parse_args()
    try:
        result = validate_run(args.run_dir, load_yaml(args.config), args.condition, args.comparison_run_dir)
        if args.write_result:
            write_result(args.run_dir, result)
    except Exception as exc:
        print(f"FAIL: validator error: {exc}", file=sys.stderr)
        return 2
    print(result["status"])
    for gate in result["gates"]:
        print(f"[{gate['status']}] {gate['name']}: {gate['detail']}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
