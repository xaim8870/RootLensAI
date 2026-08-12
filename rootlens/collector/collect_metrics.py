import argparse
import json
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
        raise RuntimeError(payload)

    return payload["data"]["result"]


def result_to_dict(results, label_name):
    output = {}

    for item in results:
        name = item["metric"].get(label_name)

        if not name:
            continue

        try:
            value = float(item["value"][1])
        except (ValueError, TypeError, IndexError):
            continue

        output[name] = output.get(name, 0.0) + value

    return output


def read_counters():
    calls_query = """
    sum by (service_name) (
      traces_span_metrics_calls_total{
        span_kind="SPAN_KIND_SERVER"
      }
    )
    """

    errors_query = """
    sum by (service_name) (
      traces_span_metrics_calls_total{
        span_kind="SPAN_KIND_SERVER",
        status_code="STATUS_CODE_ERROR"
      }
    )
    """

    duration_sum_query = """
    sum by (service_name) (
      traces_span_metrics_duration_milliseconds_sum{
        span_kind="SPAN_KIND_SERVER"
      }
    )
    """

    duration_count_query = """
    sum by (service_name) (
      traces_span_metrics_duration_milliseconds_count{
        span_kind="SPAN_KIND_SERVER"
      }
    )
    """

    return {
        "calls": result_to_dict(
            query_prometheus(calls_query),
            "service_name",
        ),
        "errors": result_to_dict(
            query_prometheus(errors_query),
            "service_name",
        ),
        "duration_sum": result_to_dict(
            query_prometheus(duration_sum_query),
            "service_name",
        ),
        "duration_count": result_to_dict(
            query_prometheus(duration_count_query),
            "service_name",
        ),
    }


def read_resources():
    cpu = result_to_dict(
        query_prometheus("container_cpu_utilization_ratio"),
        "container_name",
    )

    memory = result_to_dict(
        query_prometheus("container_memory_percent_ratio"),
        "container_name",
    )

    return cpu, memory


def safe_delta(current, previous):
    """
    Counter reset protection.

    If a container/collector restarts, a cumulative counter can become
    smaller than its previous value. In that case we do not treat the
    decrease as negative activity.
    """
    if current is None or previous is None:
        return None

    delta = current - previous

    if delta < 0:
        return None

    return delta


def build_rows(previous, current, cpu, memory, elapsed_seconds):
    now = datetime.now(timezone.utc).isoformat()
    rows = []

    for service in SERVICES:

        current_calls = current["calls"].get(service)
        previous_calls = previous["calls"].get(service)

        current_errors = current["errors"].get(service, 0.0)
        previous_errors = previous["errors"].get(service, 0.0)

        current_duration_sum = current["duration_sum"].get(service)
        previous_duration_sum = previous["duration_sum"].get(service)

        current_duration_count = current["duration_count"].get(service)
        previous_duration_count = previous["duration_count"].get(service)

        calls_delta = safe_delta(current_calls, previous_calls)

        errors_delta = safe_delta(
            current_errors,
            previous_errors,
        )

        duration_sum_delta = safe_delta(
            current_duration_sum,
            previous_duration_sum,
        )

        duration_count_delta = safe_delta(
            current_duration_count,
            previous_duration_count,
        )

        # Request rate: new server spans per second.
        if calls_delta is None or elapsed_seconds <= 0:
            request_rate = None
        else:
            request_rate = calls_delta / elapsed_seconds

        # Error rate: new error server spans per second.
        if errors_delta is None or elapsed_seconds <= 0:
            error_rate = None
        else:
            error_rate = errors_delta / elapsed_seconds

        # Mean latency of spans completed during this interval.
        if (
            duration_sum_delta is not None
            and duration_count_delta is not None
            and duration_count_delta > 0
        ):
            latency_ms = duration_sum_delta / duration_count_delta
        else:
            latency_ms = None

        rows.append(
            {
                "timestamp": now,
                "service": service,
                "request_rate": request_rate,
                "error_rate": error_rate,
                "latency_ms": latency_ms,
                "cpu": cpu.get(service),
                "memory": memory.get(service),
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-id", required=True)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--condition", default="healthy")
    parser.add_argument("--fault-service", default=None)
    parser.add_argument("--fault-type", default=None)

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_dir = root / "data" / "raw" / args.run_id

    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "run_id": args.run_id,
        "condition": args.condition,
        "fault_service": args.fault_service,
        "fault_type": args.fault_type,
        "duration_seconds": args.duration,
        "sampling_interval_seconds": args.interval,
        "prometheus_url": PROMETHEUS_URL,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "services": SERVICES,
        "feature_method": "raw_counter_deltas",
    }

    metadata_path = run_dir / "metadata.json"
    csv_path = run_dir / "metrics.csv"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Starting run: {args.run_id}")
    print(f"Duration: {args.duration}s")
    print(f"Interval: {args.interval}s")
    print("Taking initial counter snapshot...")

    previous = read_counters()
    previous_time = time.monotonic()

    all_rows = []

    # First snapshot establishes the counter baseline.
    time.sleep(args.interval)

    started = time.monotonic()

    try:
        while True:
            cycle_start = time.monotonic()

            if cycle_start - started >= args.duration:
                break

            try:
                current = read_counters()
                current_time = time.monotonic()

                elapsed_seconds = current_time - previous_time

                cpu, memory = read_resources()

                rows = build_rows(
                    previous,
                    current,
                    cpu,
                    memory,
                    elapsed_seconds,
                )

                all_rows.extend(rows)

                pd.DataFrame(all_rows).to_csv(
                    csv_path,
                    index=False,
                )

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"captured {len(rows)} services "
                    f"(window={elapsed_seconds:.2f}s, "
                    f"total rows={len(all_rows)})"
                )

                previous = current
                previous_time = current_time

            except Exception as exc:
                print(f"Collection error: {exc}")

            elapsed = time.monotonic() - cycle_start
            sleep_time = max(0, args.interval - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nCollection stopped manually.")

    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    metadata["rows_collected"] = len(all_rows)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print()
    print(f"Saved run to: {run_dir}")


if __name__ == "__main__":
    main()