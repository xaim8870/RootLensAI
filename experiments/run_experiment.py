"""Safe orchestration for paired RootLens fault and recovery experiments."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

from validate_run import load_yaml, validate_run, write_result


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = REPO_ROOT / "rootlens" / "collector" / "collect_metrics.py"
DATA_ROOT = REPO_ROOT / "rootlens" / "data" / "raw"
SUMMARY_ROOT = REPO_ROOT / "experiments" / "summaries"
ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
SERVICE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ExperimentFailure(RuntimeError):
    pass


def validate_config(config: dict[str, Any]) -> None:
    required = {"schema_version", "experiment_id", "fault_service", "fault_type", "fault_magnitude", "conditions",
                "intervention", "collection", "repetitions", "expected_services", "validation", "infrastructure"}
    missing = sorted(required - set(config))
    if missing:
        raise ExperimentFailure(f"configuration missing keys: {missing}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", str(config["experiment_id"])):
        raise ExperimentFailure("experiment_id contains unsafe characters")
    if not SERVICE_NAME_RE.fullmatch(str(config["fault_service"])):
        raise ExperimentFailure("fault_service contains unsafe characters")
    env_name = str(config["intervention"].get("environment_variable", ""))
    if not ENV_NAME_RE.fullmatch(env_name):
        raise ExperimentFailure("intervention environment variable is invalid")
    if config["fault_service"] not in config["expected_services"]:
        raise ExperimentFailure("fault_service must be in expected_services")
    for key in ("duration_seconds", "sampling_interval_seconds", "warmup_seconds"):
        if int(config["collection"][key]) < 0 or (key in ("duration_seconds", "sampling_interval_seconds") and int(config["collection"][key]) == 0):
            raise ExperimentFailure(f"collection.{key} must be positive (warm-up may be zero)")
    conditions = config["conditions"]
    if not isinstance(conditions, dict) or not all(isinstance(conditions.get(key), str) and conditions[key] for key in ("fault", "recovery")):
        raise ExperimentFailure("conditions.fault and conditions.recovery must be non-empty strings")
    verification = config["intervention"].get("verification", {})
    if verification.get("method") not in {"node_env", "shell_env"} or not verification.get("executable"):
        raise ExperimentFailure("intervention verification must configure node_env or shell_env and an executable")
    baseline = config["validation"].get("baseline_run_dir")
    if baseline and not (REPO_ROOT / baseline / "metrics.csv").is_file():
        raise ExperimentFailure(f"validation baseline dataset is missing: {baseline}")
    infra = config["infrastructure"]
    if infra.get("compose_files") != ["compose.yaml", "compose.observability.yaml"] or infra.get("env_files") != [".env", ".env.override"]:
        raise ExperimentFailure("RootLens requires both canonical Compose files and both environment files in order")


def compose_base(config: dict[str, Any]) -> list[str]:
    infra = config["infrastructure"]
    command = ["docker", "compose"]
    for env_file in infra["env_files"]:
        command += ["--env-file", env_file]
    for compose_file in infra["compose_files"]:
        command += ["-f", compose_file]
    return command


def run_command(command: list[str], cwd: Path, env: dict[str, str] | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+ " + subprocess.list2cmdline(command), flush=True)
    try:
        return subprocess.run(command, cwd=cwd, env=env, check=True, text=True,
                              capture_output=capture)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise ExperimentFailure(f"command failed: {detail.strip()}") from exc


def compose_env(config: dict[str, Any], value: str) -> dict[str, str]:
    env = os.environ.copy()
    env[str(config["intervention"]["environment_variable"])] = str(value)
    return env


def check_prometheus(config: dict[str, Any]) -> None:
    infra = config["infrastructure"]
    url = infra["prometheus_url"].rstrip("/")
    try:
        ready = requests.get(f"{url}/-/ready", timeout=10)
        ready.raise_for_status()
        response = requests.get(f"{url}/api/v1/query", params={"query": "traces_span_metrics_calls_total"}, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise ExperimentFailure(f"Prometheus prerequisite failed: {exc}") from exc
    results = payload.get("data", {}).get("result", []) if payload.get("status") == "success" else []
    if not results:
        raise ExperimentFailure("traces_span_metrics_calls_total has no current series")
    newest = max(float(item["value"][0]) for item in results)
    age = time.time() - newest
    if age > float(infra["prometheus_max_sample_age_seconds"]):
        raise ExperimentFailure(f"span metrics are stale: newest sample is {age:.1f}s old")


def check_services(config: dict[str, Any]) -> None:
    project = REPO_ROOT / config["infrastructure"]["compose_project_directory"]
    result = run_command(compose_base(config) + ["ps", "--services", "--filter", "status=running"], project, capture=True)
    running = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    required = set(config["expected_services"]) | {"otel-collector", "prometheus"}
    missing = sorted(required - running)
    if missing:
        raise ExperimentFailure(f"expected Docker services are not running: {missing}")


def set_target(config: dict[str, Any], value: str) -> None:
    project = REPO_ROOT / config["infrastructure"]["compose_project_directory"]
    service = str(config["fault_service"])
    run_command(compose_base(config) + ["up", "-d", "--no-deps", "--force-recreate", service],
                project, env=compose_env(config, value))
    variable = str(config["intervention"]["environment_variable"])
    verification = config["intervention"]["verification"]
    if verification["method"] == "node_env":
        script = f"console.log(JSON.stringify(process.env.{variable}))"
        command = ["docker", "exec", service, str(verification["executable"]), "-e", script]
        result = run_command(command, project, capture=True)
        try:
            actual = json.loads(result.stdout.strip())
        except json.JSONDecodeError as exc:
            raise ExperimentFailure(f"container environment verification returned invalid JSON: {result.stdout!r}") from exc
    else:
        script = f'printf "%s" "${variable}"'
        command = ["docker", "exec", service, str(verification["executable"]), "-c", script]
        result = run_command(command, project, capture=True)
        actual = result.stdout
    if actual != str(value):
        raise ExperimentFailure(f"container environment verification failed: expected {value!r}, got {actual!r}")


def available_sequences(config: dict[str, Any], count: int, smoke: bool = False) -> list[int]:
    experiment_id = config["experiment_id"]
    prefix = str(config["smoke"]["run_id_prefix"]) if smoke else experiment_id
    pattern = re.compile(rf"^{re.escape(prefix)}(?:_recovery)?_(\d{{3}})$")
    used = {int(match.group(1)) for path in DATA_ROOT.iterdir() if path.is_dir() and (match := pattern.fullmatch(path.name))}
    result: list[int] = []
    sequence = 1
    while len(result) < count:
        if sequence not in used:
            result.append(sequence)
        sequence += 1
    return result


def collector_command(config: dict[str, Any], run_id: str, condition: str) -> list[str]:
    collection = config["collection"]
    return [sys.executable, str(COLLECTOR), "--run-id", run_id,
            "--duration", str(collection["duration_seconds"]), "--interval", str(collection["sampling_interval_seconds"]),
            "--condition", condition, "--fault-service", str(config["fault_service"]),
            "--fault-type", str(config["fault_type"]), "--promql-window", str(collection["promql_window"]),
            "--warmup", str(collection["warmup_seconds"])]


def collect_and_validate(config: dict[str, Any], run_id: str, condition: str, comparison: Path | None = None) -> dict[str, Any]:
    run_dir = DATA_ROOT / run_id
    if run_dir.exists():
        raise ExperimentFailure(f"refusing to overwrite existing run directory: {run_dir}")
    label = str(config["conditions"][condition])
    try:
        run_command(collector_command(config, run_id, label), REPO_ROOT)
        result = validate_run(run_dir, config, condition, comparison)
        write_result(run_dir, result)
    except Exception as exc:
        validation_path = run_dir / "validation.json"
        if run_dir.is_dir() and not validation_path.exists():
            incomplete = {
                "schema_version": 1, "run_id": run_id, "condition": condition,
                "status": "FAIL", "research_valid": False,
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "gates": [{"name": "collection_or_validation_completed", "status": "FAIL", "detail": str(exc)}],
            }
            write_result(run_dir, incomplete)
        raise
    if result["status"] != "PASS":
        failures = [g["name"] for g in result["gates"] if g["status"] == "FAIL"]
        raise ExperimentFailure(f"{run_id} validation failed: {failures}")
    return result


def planned_commands(config: dict[str, Any], repetitions: int, fault_run_id: str = "<fault-run-id>", recovery_run_id: str = "<recovery-run-id>") -> list[str]:
    base = subprocess.list2cmdline(compose_base(config))
    service, variable = config["fault_service"], config["intervention"]["environment_variable"]
    verification = config["intervention"]["verification"]
    if verification["method"] == "node_env":
        verify_command = (f'docker exec {service} {verification["executable"]} -e '
                          f'"console.log(JSON.stringify(process.env.{variable}))"')
    else:
        verify_command = (f'docker exec {service} {verification["executable"]} -c '
                          f"'printf \"%s\" \"${variable}\"'")
    return [
        f"GET {config['infrastructure']['prometheus_url']}/-/ready",
        f"GET {config['infrastructure']['prometheus_url']}/api/v1/query?query=traces_span_metrics_calls_total",
        f"{base} ps --services --filter status=running",
        f'$env:{variable}="{config["intervention"]["fault_value"]}"; {base} up -d --no-deps --force-recreate {service}',
        verify_command,
        subprocess.list2cmdline(collector_command(config, fault_run_id, config["conditions"]["fault"])),
        f'$env:{variable}="{config["intervention"]["recovery_value"]}"; {base} up -d --no-deps --force-recreate {service}',
        verify_command,
        subprocess.list2cmdline(collector_command(config, recovery_run_id, config["conditions"]["recovery"])),
        f"Repeat paired sequence {repetitions} time(s); abort on first critical failure.",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--smoke", action="store_true", help="run one configured short smoke fault/recovery pair")
    parser.add_argument("--dry-run", action="store_true", help="validate and print commands without Docker, Prometheus, sleeps, or collection")
    args = parser.parse_args()
    try:
        config = load_yaml(args.config.resolve())
        validate_config(config)
        if args.smoke:
            if "smoke" not in config:
                raise ExperimentFailure("configuration does not define smoke settings")
            if args.repetitions not in (None, 1):
                raise ExperimentFailure("smoke mode always runs exactly one fault/recovery pair")
            config = copy.deepcopy(config)
            smoke_duration = int(config["smoke"]["duration_seconds"])
            if smoke_duration < 1:
                raise ExperimentFailure("smoke.duration_seconds must be positive")
            smoke_prefix = str(config["smoke"].get("run_id_prefix", ""))
            if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", smoke_prefix):
                raise ExperimentFailure("smoke.run_id_prefix contains unsafe characters")
            config["collection"]["duration_seconds"] = smoke_duration
            repetitions = 1
        else:
            repetitions = args.repetitions if args.repetitions is not None else int(config["repetitions"])
        if repetitions < 1:
            raise ExperimentFailure("repetitions must be at least 1")
        if args.dry_run:
            sequences = available_sequences(config, repetitions, args.smoke)
            if args.smoke:
                prefix = config["smoke"]["run_id_prefix"]
                fault_id = f"{prefix}_{sequences[0]:03d}"
                recovery_id = f"{prefix}_recovery_{sequences[0]:03d}"
            else:
                fault_id = recovery_id = None
            print(json.dumps({"status": "DRY_RUN", "config": str(args.config.resolve()),
                              "mode": "smoke" if args.smoke else "official", "sequences": sequences,
                              "repetitions": repetitions,
                              "commands": planned_commands(config, repetitions, fault_id or "<fault-run-id>", recovery_id or "<recovery-run-id>")}, indent=2))
            return 0

        sequences = available_sequences(config, repetitions, args.smoke)
        start_sequence = sequences[0]
        summary: dict[str, Any] = {"schema_version": 1, "experiment_id": config["experiment_id"],
                                  "started_at": datetime.now(timezone.utc).isoformat(), "status": "RUNNING", "runs": []}
        fault_active = False
        try:
            check_prometheus(config)
            check_services(config)
            for sequence in sequences:
                prefix = config["smoke"]["run_id_prefix"] if args.smoke else config["experiment_id"]
                fault_id = f"{prefix}_{sequence:03d}"
                recovery_id = f"{prefix}_recovery_{sequence:03d}"
                pair: dict[str, Any] = {"sequence": sequence, "fault_run_id": fault_id, "recovery_run_id": recovery_id}
                summary["runs"].append(pair)
                fault_active = True
                set_target(config, config["intervention"]["fault_value"])
                baseline = config["validation"].get("baseline_run_dir")
                comparison = REPO_ROOT / baseline if baseline else None
                fault_result = collect_and_validate(config, fault_id, "fault", comparison)
                pair["fault"] = fault_result
                set_target(config, config["intervention"]["recovery_value"])
                fault_active = False
                recovery_result = collect_and_validate(config, recovery_id, "recovery", DATA_ROOT / fault_id)
                pair["recovery"] = recovery_result
        except Exception as exc:
            summary["status"] = "FAIL"
            summary["error"] = str(exc)
            raise
        finally:
            if fault_active:
                try:
                    set_target(config, config["intervention"]["recovery_value"])
                    fault_active = False
                except Exception as recovery_exc:
                    summary["emergency_recovery_error"] = str(recovery_exc)
            summary["finished_at"] = datetime.now(timezone.utc).isoformat()
            if summary["status"] == "RUNNING":
                summary["status"] = "PASS"
            SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
            summary_path = SUMMARY_ROOT / f"{config['experiment_id']}_{start_sequence:03d}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(f"Summary: {summary_path}")
        return 0 if summary["status"] == "PASS" else 1
    except ExperimentFailure as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"FAIL: unexpected error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
