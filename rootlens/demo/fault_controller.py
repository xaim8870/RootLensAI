"""Allow-listed Docker fault controls for the RootLensAI live demonstration.

This module is deliberately independent of the inference package.  Its state is
presenter ground truth only and is never included in model features.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from run_experiment import (  # noqa: E402
    ExperimentFailure,
    check_prometheus,
    compose_base,
    run_command,
    validate_config,
    wait_for_workload,
    wait_service_healthy,
)
from validate_run import load_yaml  # noqa: E402


class FaultControllerError(RuntimeError):
    """A concise, UI-safe fault-controller failure."""


@dataclass(frozen=True)
class FaultSpec:
    service_key: str
    fault_type: str
    config_name: str
    display_value: str


FAULTS: dict[tuple[str, str], FaultSpec] = {
    ("payment", "latency"): FaultSpec("payment", "latency", "payment_latency_500.yaml", "+500 ms"),
    ("cart", "latency"): FaultSpec("cart", "latency", "cart_latency_500.yaml", "+500 ms"),
    ("checkout", "latency"): FaultSpec("checkout", "latency", "checkout_latency_500.yaml", "+500 ms"),
    ("checkout", "error"): FaultSpec("checkout", "error", "checkout_error_50.yaml", "50%"),
    ("product_catalog", "latency"): FaultSpec("product_catalog", "latency", "product_catalog_latency_500.yaml", "+500 ms"),
    ("product_catalog", "error"): FaultSpec("product_catalog", "error", "product_catalog_error_50.yaml", "50%"),
}

_CONFIG_ROOT = REPO_ROOT / "experiments" / "configs"
_STATE_PATH = REPO_ROOT / "rootlens" / "data" / "runtime" / "demo_fault_state.json"
_LOCK = threading.Lock()


def supported_faults() -> dict[str, list[str]]:
    """Return a stable UI allow-list; no command input is accepted."""
    result: dict[str, list[str]] = {}
    for service, fault_type in FAULTS:
        result.setdefault(service, []).append(fault_type)
    return result


def _load_config(spec: FaultSpec) -> dict[str, Any]:
    config = load_yaml(_CONFIG_ROOT / spec.config_name)
    validate_config(config)
    return config


def _all_configs() -> list[dict[str, Any]]:
    return [_load_config(spec) for spec in FAULTS.values()]


def _controls_by_service() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for config in _all_configs():
        variable = str(config["intervention"]["environment_variable"])
        if variable in seen:
            continue
        seen.add(variable)
        grouped.setdefault(str(config["fault_service"]), []).append(config)
    return grouped


def _verify(config: dict[str, Any], expected: str) -> None:
    project = REPO_ROOT / config["infrastructure"]["compose_project_directory"]
    service = str(config["fault_service"])
    variable = str(config["intervention"]["environment_variable"])
    verification = config["intervention"]["verification"]
    method = verification["method"]
    if method == "node_env":
        script = f"console.log(JSON.stringify(process.env.{variable}))"
        command = ["docker", "exec", service, str(verification["executable"]), "-e", script]
        result = run_command(command, project, capture=True)
        try:
            actual = json.loads(result.stdout.strip())
        except json.JSONDecodeError as exc:
            raise FaultControllerError(f"Invalid environment response from {service}") from exc
    elif method == "shell_env":
        command = ["docker", "exec", service, str(verification["executable"]), "-c", f'printf "%s" "${variable}"']
        actual = run_command(command, project, capture=True).stdout
    else:
        arguments = [str(value).replace("{environment_variable}", variable)
                     for value in verification.get("arguments", [])]
        command = ["docker", "exec", service, str(verification["executable"]), *arguments]
        actual = run_command(command, project, capture=True).stdout
    if actual != expected:
        raise FaultControllerError(
            f"{service} verification failed for {variable}: expected {expected!r}, got {actual!r}"
        )


def _apply_service(service: str, configs: list[dict[str, Any]], values: dict[str, str]) -> None:
    reference = configs[0]
    project = REPO_ROOT / reference["infrastructure"]["compose_project_directory"]
    environment = os.environ.copy()
    for config in configs:
        variable = str(config["intervention"]["environment_variable"])
        environment[variable] = values.get(variable, str(config["intervention"]["recovery_value"]))
    run_command(
        compose_base(reference) + ["up", "-d", "--no-deps", "--force-recreate", service],
        project,
        env=environment,
    )
    wait_service_healthy(reference, service, timeout_seconds=90)
    for config in configs:
        variable = str(config["intervention"]["environment_variable"])
        _verify(config, environment[variable])


def _write_state(state: dict[str, Any]) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = _STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(_STATE_PATH)


def get_fault_state() -> dict[str, Any]:
    if not _STATE_PATH.is_file():
        return {"active": False, "service": None, "fault_type": None, "display_value": None}
    try:
        state = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"active": False, "service": None, "fault_type": None, "display_value": None}
    return state if state.get("active") else {
        "active": False, "service": None, "fault_type": None, "display_value": None
    }


def inject_fault(service: str, fault_type: str) -> dict[str, Any]:
    """Apply one supported fault and verify its exact runtime value."""
    spec = FAULTS.get((service, fault_type))
    if spec is None:
        raise FaultControllerError(f"Unsupported fault selection: {service}/{fault_type}")
    with _LOCK:
        try:
            grouped = _controls_by_service()
            config = _load_config(spec)
            current = get_fault_state()
            if current["active"] and (
                current["service"] != service or current["fault_type"] != fault_type
            ):
                raise FaultControllerError(
                    "A different demo fault is active; restore the system before injecting another fault"
                )
            selected_docker_service = str(config["fault_service"])
            for other_service, other_configs in grouped.items():
                if other_service == selected_docker_service:
                    continue
                for other_config in other_configs:
                    _verify(other_config, str(other_config["intervention"]["recovery_value"]))
            check_prometheus(config)
            variable = str(config["intervention"]["environment_variable"])
            values = {variable: str(config["intervention"]["fault_value"])}
            _apply_service(selected_docker_service, grouped[selected_docker_service], values)
            check_prometheus(config)
        except (ExperimentFailure, OSError) as exc:
            raise FaultControllerError(str(exc)) from exc
        state = {
            "active": True,
            "service": service,
            "docker_service": config["fault_service"],
            "fault_type": fault_type,
            "display_value": spec.display_value,
            "environment_variable": variable,
            "runtime_value": str(config["intervention"]["fault_value"]),
            "changed_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_state(state)
        return state


def restore_fault() -> dict[str, Any]:
    """Idempotently restore every demo control to its frozen healthy value."""
    with _LOCK:
        try:
            grouped = _controls_by_service()
            for service, configs in grouped.items():
                _apply_service(service, configs, {})
                readiness_config = next((config for config in configs if config.get("recovery_readiness")), None)
                if readiness_config is not None:
                    wait_for_workload(readiness_config)
            check_prometheus(next(iter(grouped.values()))[0])
        except (ExperimentFailure, OSError) as exc:
            raise FaultControllerError(str(exc)) from exc
        state = {
            "active": False,
            "service": None,
            "fault_type": None,
            "display_value": None,
            "restored_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_state(state)
        return state
