"""Thin FastAPI adapter around the existing RootLens runtime functions."""

from __future__ import annotations

import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import requests
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rootlens.api.runtime import (
    INJECT_RCA_LOCK_SECONDS,
    RESTORE_RCA_LOCK_SECONDS,
    runtime_state,
    utc_now,
)
from rootlens.demo.fault_controller import (
    FaultControllerError,
    get_fault_state,
    inject_fault,
    restore_fault,
    supported_faults,
)
LOGGER = logging.getLogger("rootlens.api")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMETHEUS_URL = "http://localhost:9090"
CALIBRATION_PATH = (
    REPO_ROOT
    / "rootlens/data/reports/graphsage_rca_v2_calibration"
    / "validation_calibration_metrics.json"
)
TOPOLOGY_PATH = REPO_ROOT / "rootlens/config/service_graph_v1.yaml"


class FaultRequest(BaseModel):
    service: str
    fault_type: Literal["latency", "error"]


app = FastAPI(
    title="RootLensAI API",
    version="1.0.0",
    description="Live telemetry, GraphSAGE RCA, safe fault control, and grounded RAG.",
)

allowed_origins = [
    item.strip()
    for item in os.getenv(
        "ROOTLENS_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _prometheus_ready() -> tuple[bool, str | None]:
    url = os.getenv("ROOTLENS_PROMETHEUS_URL", DEFAULT_PROMETHEUS_URL).rstrip("/")
    try:
        response = requests.get(
            f"{url}/api/v1/query",
            params={"query": "count(traces_span_metrics_calls_total)"},
            timeout=3,
        )
        response.raise_for_status()
        payload = response.json()
        values = payload.get("data", {}).get("result", [])
        return payload.get("status") == "success" and bool(values), None
    except (requests.RequestException, ValueError) as exc:
        return False, str(exc)


def _fault_payload() -> dict[str, Any]:
    return {
        "fault": get_fault_state(),
        "supported_faults": supported_faults(),
        **runtime_state.snapshot(),
    }


@lru_cache(maxsize=1)
def _rca_engine() -> Any:
    """Load the canonical model once per API process, like Streamlit's cache."""
    from rootlens.inference.rca_inference import RCAInference

    return RCAInference(device="cpu")


@app.get("/api/health")
def health() -> dict[str, Any]:
    connected, error = _prometheus_ready()
    return {
        "status": "ok",
        "api_ready": True,
        "prometheus_connected": connected,
        "prometheus_error": error,
        "timestamp": utc_now(),
    }


@app.get("/api/rca")
def refresh_rca() -> dict[str, Any]:
    remaining = runtime_state.gate_remaining()
    if remaining:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "telemetry_gate_active",
                "message": "RCA refresh is locked while telemetry stabilizes.",
                "remaining_seconds": remaining,
            },
        )

    if not runtime_state.operation_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A testbed operation is running")
    started = time.perf_counter()
    try:
        from rootlens.inference.live_telemetry import run_live_rca

        result = run_live_rca(
            demo_mode=False,
            device="cpu",
            inference=_rca_engine(),
        )
    except Exception as exc:
        LOGGER.exception("Live RCA failed")
        with runtime_state.lock:
            runtime_state.last_rca_duration = time.perf_counter() - started
        raise HTTPException(status_code=503, detail=f"RCA unavailable: {exc}") from exc
    finally:
        runtime_state.operation_lock.release()

    with runtime_state.lock:
        runtime_state.rca_result = result
        runtime_state.rca_stale = False
        runtime_state.last_rca_duration = time.perf_counter() - started
        runtime_state.rag_analysis = None
        runtime_state.rag_source_timestamp = None
        runtime_state.last_rag_duration = None
    return result


@app.post("/api/fault/inject")
def apply_fault(request: FaultRequest) -> dict[str, Any]:
    choices = supported_faults()
    if request.service not in choices or request.fault_type not in choices[request.service]:
        raise HTTPException(status_code=422, detail="Unsupported allow-listed fault selection")
    if not runtime_state.operation_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another testbed operation is running")

    started = time.perf_counter()
    runtime_state.invalidate(
        seconds=INJECT_RCA_LOCK_SECONDS,
        reason="Testbed change in progress; telemetry stabilization follows",
    )
    try:
        state = inject_fault(request.service, request.fault_type)
        duration = time.perf_counter() - started
        with runtime_state.lock:
            runtime_state.last_operation_type = (
                f"Injected {request.service.replace('_', ' ').title()} "
                f"{request.fault_type.title()}"
            )
            runtime_state.last_operation_duration = duration
            runtime_state.last_operation_finished_at = utc_now()
        runtime_state.invalidate(
            seconds=INJECT_RCA_LOCK_SECONDS,
            reason=(
                "Fault injected; waiting 30s for the PromQL window to reflect "
                "the new condition"
            ),
        )
        return {"status": "ok", "fault": state, "runtime": runtime_state.snapshot()}
    except FaultControllerError as exc:
        with runtime_state.lock:
            runtime_state.last_operation_type = "Fault injection failed"
            runtime_state.last_operation_duration = time.perf_counter() - started
            runtime_state.last_operation_finished_at = utc_now()
        LOGGER.exception("Fault injection failed")
        raise HTTPException(status_code=503, detail=f"Fault injection failed: {exc}") from exc
    finally:
        runtime_state.operation_lock.release()


@app.post("/api/fault/restore")
def restore_system() -> dict[str, Any]:
    if not runtime_state.operation_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another testbed operation is running")

    started = time.perf_counter()
    runtime_state.invalidate(
        seconds=RESTORE_RCA_LOCK_SECONDS,
        reason="System restoration in progress; telemetry recovery follows",
    )
    try:
        state = restore_fault()
        duration = time.perf_counter() - started
        with runtime_state.lock:
            runtime_state.last_operation_type = "System restore"
            runtime_state.last_operation_duration = duration
            runtime_state.last_operation_finished_at = utc_now()
        runtime_state.invalidate(
            seconds=RESTORE_RCA_LOCK_SECONDS,
            reason=(
                "System restored; waiting 60s for fault residue to leave the "
                "rolling telemetry window"
            ),
        )
        return {"status": "ok", "fault": state, "runtime": runtime_state.snapshot()}
    except FaultControllerError as exc:
        with runtime_state.lock:
            runtime_state.last_operation_type = "System restore failed"
            runtime_state.last_operation_duration = time.perf_counter() - started
            runtime_state.last_operation_finished_at = utc_now()
        LOGGER.exception("System restoration failed")
        raise HTTPException(status_code=503, detail=f"System restoration failed: {exc}") from exc
    finally:
        runtime_state.operation_lock.release()


@app.get("/api/fault/state")
def fault_state() -> dict[str, Any]:
    return _fault_payload()


@app.post("/api/rag/analyze")
def analyze_current_rca() -> dict[str, Any]:
    with runtime_state.lock:
        result = runtime_state.rca_result
        stale = runtime_state.rca_stale
    if result is None or stale:
        raise HTTPException(
            status_code=409,
            detail="A fresh RCA result is required before AI analysis",
        )

    started = time.perf_counter()
    try:
        from rootlens.rag.explainability import generate_rca_explanation

        analysis = generate_rca_explanation(result)
    except Exception as exc:
        LOGGER.exception("RAG analysis failed")
        raise HTTPException(status_code=503, detail=f"AI analysis unavailable: {exc}") from exc

    with runtime_state.lock:
        if runtime_state.rca_result is not result or runtime_state.rca_stale:
            raise HTTPException(
                status_code=409,
                detail="The RCA result changed while AI analysis was running",
            )
        runtime_state.rag_analysis = analysis
        runtime_state.rag_source_timestamp = str(result["timestamp"])
        runtime_state.last_rag_duration = time.perf_counter() - started
    return analysis


@app.get("/api/model/calibration")
def model_calibration() -> dict[str, Any]:
    if not CALIBRATION_PATH.is_file():
        raise HTTPException(status_code=503, detail="Calibration artifact is unavailable")
    try:
        source = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Calibration artifact is invalid") from exc
    return {
        "validation_accuracy": source["validation_accuracy"],
        "mean_confidence": source["validation_mean_confidence"],
        "accuracy_confidence_gap": source["validation_accuracy_confidence_gap"],
        "ece": source["validation_ece"],
        "nll": source["validation_nll"],
        "brier_score": source["validation_brier_score"],
        "scope": "Dataset v2 validation",
        "test_set_evaluated": bool(source["sealed_test_evaluated"]),
        "source": str(CALIBRATION_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
    }


@app.get("/api/topology")
def topology() -> dict[str, Any]:
    try:
        source = yaml.safe_load(TOPOLOGY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise HTTPException(status_code=503, detail="Frozen topology is unavailable") from exc
    return {
        "node_order": source["node_order"],
        "edges": source["edges"],
        "expected_counts": source["expected_counts"],
        "version": source["graph"]["version"],
    }
