import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


PROMETHEUS_URL = "http://localhost:9090"

SERVICES = [
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


def query_prometheus(query: str):
    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": query},
        timeout=10,
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("status") != "success":
        raise RuntimeError(
            f"Prometheus query failed: {payload}"
        )

    return payload["data"]["result"]


def result_to_dict(results, label_name):
    """
    Convert a Prometheus instant-vector result into:

        {
            "frontend": 1.23,
            "cart": 0.45,
            ...
        }

    Non-finite values such as NaN are represented as None.
    """
    output = {}

    for item in results:
        name = item.get("metric", {}).get(label_name)

        if not name:
            continue

        try:
            value = float(item["value"][1])
        except (ValueError, TypeError, IndexError, KeyError):
            continue

        if not math.isfinite(value):
            output[name] = None
            continue

        output[name] = value

    return output


def build_queries(window: str):
    """
    Build the exact PromQL feature definitions used by RootLens.

    Only SERVER spans are used for service-level RED metrics.

    EventStream spans are excluded from latency because they represent
    long-lived streaming operations rather than ordinary request/response
    service latency.
    """

    request_rate = f"""
    sum by (service_name) (
      rate(
        traces_span_metrics_calls_total{{
          span_kind="SPAN_KIND_SERVER"
        }}[{window}]
      )
    )
    """

    error_rps = f"""
    sum by (service_name) (
      rate(
        traces_span_metrics_calls_total{{
          span_kind="SPAN_KIND_SERVER",
          status_code="STATUS_CODE_ERROR"
        }}[{window}]
      )
    )
    """

    error_rate = f"""
    (
      sum by (service_name) (
        rate(
          traces_span_metrics_calls_total{{
            span_kind="SPAN_KIND_SERVER",
            status_code="STATUS_CODE_ERROR"
          }}[{window}]
        )
      )
    )
    /
    clamp_min(
      sum by (service_name) (
        rate(
          traces_span_metrics_calls_total{{
            span_kind="SPAN_KIND_SERVER"
          }}[{window}]
        )
      ),
      1e-9
    )
    """

    latency_ms = f"""
    sum by (service_name) (
      rate(
        traces_span_metrics_duration_milliseconds_sum{{
          span_kind="SPAN_KIND_SERVER",
          span_name!~".*EventStream.*"
        }}[{window}]
      )
    )
    /
    sum by (service_name) (
      rate(
        traces_span_metrics_duration_milliseconds_count{{
          span_kind="SPAN_KIND_SERVER",
          span_name!~".*EventStream.*"
        }}[{window}]
      )
    )
    """

    return {
        "request_rate": request_rate,
        "error_rps": error_rps,
        "error_rate": error_rate,
        "latency_ms": latency_ms,
    }


def read_red_metrics(queries):
    """
    Read service-level RED metrics already calculated by Prometheus.

    Python does NOT calculate cumulative counter differences anymore.
    """

    return {
        "request_rate": result_to_dict(
            query_prometheus(queries["request_rate"]),
            "service_name",
        ),
        "error_rps": result_to_dict(
            query_prometheus(queries["error_rps"]),
            "service_name",
        ),
        "error_rate": result_to_dict(
            query_prometheus(queries["error_rate"]),
            "service_name",
        ),
        "latency_ms": result_to_dict(
            query_prometheus(queries["latency_ms"]),
            "service_name",
        ),
    }


def read_resources():
    """
    CPU and memory are gauges, so they do not need rate/delta logic.
    """

    cpu = result_to_dict(
        query_prometheus(
            "container_cpu_utilization_ratio"
        ),
        "container_name",
    )

    memory = result_to_dict(
        query_prometheus(
            "container_memory_percent_ratio"
        ),
        "container_name",
    )

    return cpu, memory


def build_rows(red, cpu, memory):
    now = datetime.now(timezone.utc).isoformat()
    rows = []

    for service in SERVICES:
        request_rate = red["request_rate"].get(service)

        # Missing request-rate telemetry is different from an observed
        # rate of zero.
        if request_rate is None:
            has_requests = None
            error_rate = None
            error_rps = None
            latency_ms = None

        elif request_rate <= 1e-12:
            # Service is observed, but no requests occurred in the
            # PromQL lookback window.
            has_requests = 0
            request_rate = 0.0
            error_rate = None
            error_rps = 0.0
            latency_ms = None

        else:
            has_requests = 1

            # If request traffic exists but no ERROR-labelled series
            # exists, the natural error rate is zero.
            error_rate = red["error_rate"].get(service)

            if error_rate is None:
                error_rate = 0.0

            error_rps = red["error_rps"].get(service)

            if error_rps is None:
                error_rps = 0.0

            latency_ms = red["latency_ms"].get(service)

        rows.append(
            {
                "timestamp": now,
                "service": service,

                # Model features
                "request_rate": request_rate,
                "error_rate": error_rate,
                "latency_ms": latency_ms,
                "cpu": cpu.get(service),
                "memory": memory.get(service),

                # Diagnostic / missingness fields
                "error_rps": error_rps,
                "has_requests": has_requests,
            }
        )

    return rows


def validate_rows(rows):
    """
    Lightweight engineering validation.

    We warn about impossible values but do not discard observations.
    High latency is deliberately NOT capped because future RootLens
    faults may legitimately cause very large latency.
    """

    problems = []

    for row in rows:
        service = row["service"]

        request_rate = row["request_rate"]
        error_rate = row["error_rate"]
        latency_ms = row["latency_ms"]

        if request_rate is not None and request_rate < 0:
            problems.append(
                f"{service}: negative request_rate={request_rate}"
            )

        if error_rate is not None:
            if error_rate < 0 or error_rate > 1:
                problems.append(
                    f"{service}: invalid error_rate={error_rate}"
                )

        if latency_ms is not None and latency_ms < 0:
            problems.append(
                f"{service}: negative latency_ms={latency_ms}"
            )

    return problems


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-id", required=True)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--condition", default="healthy")
    parser.add_argument("--fault-service", default=None)
    parser.add_argument("--fault-type", default=None)

    parser.add_argument(
        "--promql-window",
        default="30s",
        help=(
            "Prometheus lookback window used for RED metrics. "
            "Examples: 30s, 1m, 2m. Default: 30s"
        ),
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help=(
            "Seconds to wait before recording so PromQL has recent "
            "telemetry history. Default: 30"
        ),
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_dir = root / "data" / "raw" / args.run_id

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    queries = build_queries(args.promql_window)

    metadata = {
        "run_id": args.run_id,
        "condition": args.condition,
        "fault_service": args.fault_service,
        "fault_type": args.fault_type,

        "duration_seconds": args.duration,
        "sampling_interval_seconds": args.interval,
        "warmup_seconds": args.warmup,

        "prometheus_url": PROMETHEUS_URL,
        "promql_window": args.promql_window,

        "started_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "services": SERVICES,

        "feature_method": "promql_rate",

        "feature_definitions": {
            "request_rate": (
                "SERVER span calls per second"
            ),
            "error_rate": (
                "ERROR SERVER span rate / all SERVER span rate"
            ),
            "latency_ms": (
                "mean SERVER span duration from rate(sum)/rate(count), "
                "excluding EventStream spans"
            ),
            "cpu": "container_cpu_utilization_ratio",
            "memory": "container_memory_percent_ratio",
        },

        "diagnostic_fields": [
            "error_rps",
            "has_requests",
        ],

        "promql_queries": queries,
    }

    metadata_path = run_dir / "metadata.json"
    csv_path = run_dir / "metrics.csv"

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(f"Starting run: {args.run_id}")
    print(f"Duration: {args.duration}s")
    print(f"Interval: {args.interval}s")
    print(f"PromQL window: {args.promql_window}")
    print(f"Warm-up: {args.warmup}s")

    if args.warmup > 0:
        print(
            f"Waiting {args.warmup}s for telemetry warm-up..."
        )
        time.sleep(args.warmup)

    all_rows = []

    started = time.monotonic()

    try:
        while True:
            cycle_start = time.monotonic()

            if cycle_start - started >= args.duration:
                break

            try:
                red = read_red_metrics(queries)

                cpu, memory = read_resources()

                rows = build_rows(
                    red,
                    cpu,
                    memory,
                )

                problems = validate_rows(rows)

                for problem in problems:
                    print(
                        f"WARNING: {problem}"
                    )

                all_rows.extend(rows)

                pd.DataFrame(all_rows).to_csv(
                    csv_path,
                    index=False,
                )

                active_services = sum(
                    1
                    for row in rows
                    if row["has_requests"] == 1
                )

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"captured {len(rows)} services "
                    f"(active={active_services}, "
                    f"total rows={len(all_rows)})"
                )

            except Exception as exc:
                print(
                    f"Collection error: {exc}"
                )

            elapsed = (
                time.monotonic()
                - cycle_start
            )

            sleep_time = max(
                0,
                args.interval - elapsed,
            )

            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(
            "\nCollection stopped manually."
        )

    metadata["finished_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    metadata["rows_collected"] = len(
        all_rows
    )

    metadata["windows_collected"] = (
        len(all_rows) // len(SERVICES)
        if SERVICES
        else 0
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print(
        f"Saved run to: {run_dir}"
    )


if __name__ == "__main__":
    main()