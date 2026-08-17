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


def _active_fraction_above_zero(df: pd.DataFrame, service: str, column: str) -> float:
    rows = df[(df["service"] == service) & (df["has_requests"] == 1)]
    if rows.empty:
        return 0.0
    return float((pd.to_numeric(rows[column], errors="coerce").fillna(0) > 0).mean())


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
    fault_type = str(config.get("fault_type"))
    if condition == "healthy":
        inactive_services: dict[str, float] = {}
        minimum_activity = float(limits["minimum_healthy_service_active_fraction"])
        for service in expected_services:
            rows = df[df["service"] == service]
            fraction = float((rows["has_requests"] == 1).mean()) if len(rows) else 0.0
            if fraction < minimum_activity:
                inactive_services[service] = fraction
        gates.append(Gate("healthy_service_activity", not inactive_services,
                          f"below_minimum={inactive_services}, minimum={minimum_activity}"))

        maximum_median = float(limits["maximum_healthy_median_error_rate"])
        excessive_medians = {}
        excessive_streaks = {}
        maximum_streak = int(limits["maximum_healthy_consecutive_error_windows"])
        for service in expected_services:
            rows = df[df["service"] == service].sort_values("timestamp")
            values = pd.to_numeric(rows["error_rate"], errors="coerce").fillna(0)
            median = float(values.median()) if len(values) else 0.0
            if median > maximum_median:
                excessive_medians[service] = median
            streak = longest = 0
            for value in values:
                streak = streak + 1 if value > 0 else 0
                longest = max(longest, streak)
            if longest > maximum_streak:
                excessive_streaks[service] = longest
        gates.append(Gate("healthy_error_level", not excessive_medians,
                          f"excessive_service_medians={excessive_medians}, maximum={maximum_median}"))
        gates.append(Gate("healthy_sustained_errors", not excessive_streaks,
                          f"excessive_consecutive_windows={excessive_streaks}, maximum={maximum_streak}"))
        metadata_ok = (str(metadata.get("condition")) == "healthy"
                       and metadata.get("fault_service") in (None, "none")
                       and metadata.get("fault_type") in (None, "none"))
        gates.append(Gate("healthy_metadata", metadata_ok,
                          f"condition={metadata.get('condition')}, fault_service={metadata.get('fault_service')}, fault_type={metadata.get('fault_type')}"))
    elif condition == "fault" and fault_type == "resource_cpu":
        target_cpu = _median(df, target, "cpu")
        baseline_cpu = None
        increase = None
        nontarget_increases: dict[str, float] = {}
        if comparison_run_dir:
            comparison = pd.read_csv(comparison_run_dir / "metrics.csv")
            baseline_cpu = _median(comparison, target, "cpu")
            if target_cpu is not None and baseline_cpu is not None:
                increase = target_cpu - baseline_cpu
            for service in config["expected_services"]:
                if service == target:
                    continue
                current = _median(df, service, "cpu")
                baseline_value = _median(comparison, service, "cpu")
                if current is not None and baseline_value is not None:
                    nontarget_increases[service] = current - baseline_value
        cpu_rows = pd.to_numeric(target_rows["cpu"], errors="coerce")
        consistent_fraction = float((cpu_rows >= float(limits["fault_cpu_consistency_threshold"])).mean()) if len(cpu_rows) else 0.0
        maximum_nontarget = max(nontarget_increases.values(), default=0.0)
        gates.append(Gate("resource_intervention_visible",
                          target_cpu is not None and target_cpu >= float(limits["minimum_fault_target_cpu"]),
                          f"target_median_cpu={target_cpu}, minimum={limits['minimum_fault_target_cpu']}"))
        gates.append(Gate("resource_increase",
                          increase is not None and increase >= float(limits["minimum_fault_cpu_increase"]),
                          f"increase={increase}, minimum={limits['minimum_fault_cpu_increase']}, baseline={baseline_cpu}"))
        gates.append(Gate("resource_consistency",
                          consistent_fraction >= float(limits["minimum_fault_cpu_window_fraction"]),
                          f"window_fraction={consistent_fraction}, minimum={limits['minimum_fault_cpu_window_fraction']}, threshold={limits['fault_cpu_consistency_threshold']}"))
        gates.append(Gate("nontarget_cpu_contamination",
                          maximum_nontarget <= float(limits["maximum_nontarget_median_cpu_increase"]),
                          f"maximum_nontarget_median_cpu_increase={maximum_nontarget}, maximum={limits['maximum_nontarget_median_cpu_increase']}, by_service={nontarget_increases}"))
    elif condition == "fault" and fault_type == "error":
        target_error_rate = _median(df, target, "error_rate")
        target_error_rps = _median(df, target, "error_rps")
        error_window_fraction = _active_fraction_above_zero(df, target, "error_rate")
        gates.append(Gate("error_intervention_visible",
                          target_error_rate is not None and target_error_rate >= float(limits["minimum_fault_target_error_rate"]),
                          f"target_median_error_rate={target_error_rate}, minimum={limits['minimum_fault_target_error_rate']}"))
        gates.append(Gate("error_activity",
                          target_error_rps is not None and target_error_rps >= float(limits["minimum_fault_target_error_rps"]),
                          f"target_median_error_rps={target_error_rps}, minimum={limits['minimum_fault_target_error_rps']}"))
        gates.append(Gate("error_window_fraction",
                          error_window_fraction >= float(limits["minimum_fault_error_window_fraction"]),
                          f"active_error_window_fraction={error_window_fraction}, minimum={limits['minimum_fault_error_window_fraction']}"))
    elif condition == "fault":
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
    elif condition == "recovery" and fault_type == "resource_cpu":
        if comparison_run_dir is None:
            gates.append(Gate("recovery_comparison", False, "fault comparison run is required"))
        else:
            fault_df = pd.read_csv(comparison_run_dir / "metrics.csv")
            recovery_cpu = _median(df, target, "cpu")
            fault_cpu = _median(fault_df, target, "cpu")
            ratio = None if recovery_cpu is None or not fault_cpu else recovery_cpu / fault_cpu
            gates.append(Gate("recovery_resource_level",
                              recovery_cpu is not None and recovery_cpu <= float(limits["maximum_recovery_target_cpu"]),
                              f"recovery_median_cpu={recovery_cpu}, maximum={limits['maximum_recovery_target_cpu']}"))
            gates.append(Gate("recovery_resource_direction",
                              ratio is not None and ratio <= float(limits["maximum_recovery_to_fault_cpu_ratio"]),
                              f"recovery_to_fault_cpu_ratio={ratio}, maximum={limits['maximum_recovery_to_fault_cpu_ratio']}, recovery={recovery_cpu}, fault={fault_cpu}"))
    elif condition == "recovery" and fault_type == "error":
        if comparison_run_dir is None:
            gates.append(Gate("recovery_comparison", False, "fault comparison run is required"))
        else:
            fault_df = pd.read_csv(comparison_run_dir / "metrics.csv")
            recovery_error = _median(df, target, "error_rate")
            fault_error = _median(fault_df, target, "error_rate")
            ratio = None if recovery_error is None or not fault_error else recovery_error / fault_error
            maximum_error = float(limits["maximum_recovery_target_error_rate"])
            maximum_ratio = float(limits["maximum_recovery_to_fault_error_ratio"])
            gates.append(Gate("recovery_error_level", recovery_error is not None and recovery_error <= maximum_error,
                              f"recovery_median_error_rate={recovery_error}, maximum={maximum_error}"))
            gates.append(Gate("recovery_error_direction", ratio is not None and ratio <= maximum_ratio,
                              f"recovery_to_fault_error_ratio={ratio}, maximum={maximum_ratio}, recovery={recovery_error}, fault={fault_error}"))
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
    parser.add_argument("--condition", required=True, choices=("fault", "recovery", "healthy"))
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
