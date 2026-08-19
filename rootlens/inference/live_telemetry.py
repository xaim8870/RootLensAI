#rootlens/inference/live_telemetry.py
#!/usr/bin/env python3
"""Prometheus-to-GraphSAGE adapter using frozen RootLens PromQL semantics."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import requests

from rootlens.collector.collect_metrics import SERVICES, build_queries, result_to_dict
from rootlens.inference.rca_inference import RCAInference


DEFAULT_PROMETHEUS_URL = "http://localhost:9090"
DEFAULT_PROMQL_WINDOW = "30s"


class PrometheusUnavailableError(RuntimeError):
    """Prometheus could not provide a complete live telemetry window."""


def _query(prometheus_url: str, query: str, timeout: float) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            f"{prometheus_url.rstrip('/')}/api/v1/query",
            params={"query": query},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise PrometheusUnavailableError(f"Prometheus query failed: {exc}") from exc
    if payload.get("status") != "success":
        raise PrometheusUnavailableError(
            f"Prometheus returned non-success status: {payload.get('error', payload)}"
        )
    return list(payload.get("data", {}).get("result", []))


def collect_live_window(
    prometheus_url: str = DEFAULT_PROMETHEUS_URL,
    promql_window: str = DEFAULT_PROMQL_WINDOW,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Collect one 12-service window with the collector's exact feature queries."""
    queries = build_queries(promql_window)
    red = {
        metric: result_to_dict(
            _query(prometheus_url, query, timeout),
            "service_name",
        )
        for metric, query in queries.items()
    }
    cpu = result_to_dict(
        _query(prometheus_url, "container_cpu_utilization_ratio", timeout),
        "container_name",
    )
    memory = result_to_dict(
        _query(prometheus_url, "container_memory_percent_ratio", timeout),
        "container_name",
    )

    metrics: dict[str, dict[str, float | int | None]] = {}
    for service in SERVICES:
        request_rate = red["request_rate"].get(service)
        if request_rate is None:
            has_requests = None
            error_rate = None
            error_rps = None
            latency_ms = None
        elif request_rate <= 1e-12:
            request_rate = 0.0
            has_requests = 0
            error_rate = None
            error_rps = 0.0
            latency_ms = None
        else:
            has_requests = 1
            error_rate = red["error_rate"].get(service)
            if error_rate is None:
                error_rate = 0.0
            error_rps = red["error_rps"].get(service)
            if error_rps is None:
                error_rps = 0.0
            latency_ms = red["latency_ms"].get(service)
        metrics[service] = {
            "cpu": cpu.get(service),
            "memory": memory.get(service),
            "request_rate": request_rate,
            "has_requests": has_requests,
            "latency_ms": latency_ms,
            "error_rps": error_rps,
            "error_rate": error_rate,
        }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prometheus_url": prometheus_url,
        "promql_window": promql_window,
        "service_metrics": metrics,
    }


def build_demo_window() -> dict[str, Any]:
    """Manually constructed, non-dataset telemetry for UI/inference testing."""
    services = [
        "frontend", "frontend-proxy", "checkout", "payment", "cart",
        "currency", "shipping", "product-catalog", "recommendation",
        "email", "ad", "quote",
    ]
    request_rates = {
        "frontend": 2.8, "frontend-proxy": 3.0, "checkout": 0.8,
        "payment": 0.35, "cart": 1.1, "currency": 1.4,
        "shipping": 0.7, "product-catalog": 2.2, "recommendation": 0.9,
        "email": 0.3, "ad": 0.8, "quote": 0.5,
    }
    metrics = {}
    for index, service in enumerate(services):
        rate = request_rates[service]
        metrics[service] = {
            "cpu": 3.0 + (index % 4) * 0.8,
            "memory": 38.0 + (index % 5) * 4.0,
            "request_rate": rate,
            "has_requests": 1,
            "latency_ms": 28.0 + (index % 3) * 7.0,
            "error_rps": 0.0,
            "error_rate": 0.0,
        }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prometheus_url": None,
        "promql_window": DEFAULT_PROMQL_WINDOW,
        "service_metrics": metrics,
        "demo": True,
    }


def run_live_rca(
    *,
    demo_mode: bool = False,
    prometheus_url: str | None = None,
    device: str = "cpu",
    inference: RCAInference | None = None,
) -> dict[str, Any]:
    """Collect or construct one window and return a UI-ready RCA result."""
    engine = inference or RCAInference(device=device)
    if demo_mode:
        window = build_demo_window()
    else:
        url = prometheus_url or os.getenv(
            "ROOTLENS_PROMETHEUS_URL", DEFAULT_PROMETHEUS_URL
        )
        window = collect_live_window(prometheus_url=url)
    prediction = engine.predict(window["service_metrics"])
    predicted = prediction["predicted_class"]
    return {
        "timestamp": window["timestamp"],
        "system_status": "healthy" if predicted == "healthy" else "incident",
        "predicted_root_cause": predicted,
        "confidence": prediction["confidence"],
        "probabilities": prediction["probabilities"],
        "service_metrics": window["service_metrics"],
        "graph_shape": prediction["graph_shape"],
        "device": prediction["device"],
        "demo_mode": demo_mode,
        "promql_window": window["promql_window"],
    }
