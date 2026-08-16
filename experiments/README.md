# RootLens experiment orchestration

This directory runs paired fault/recovery experiments against the OpenTelemetry Demo while leaving the collector, PromQL definitions, workloads, service list, and prior datasets unchanged.

## Safety model

- Every Docker Compose command uses `.env`, `.env.override`, `compose.yaml`, and `compose.observability.yaml`.
- Fault changes are supplied to Compose as a process environment override; no source or environment file is edited.
- Container environment values are read with the service-aware verifier configured for the experiment. Payment uses `/nodejs/bin/node` with JSON output; Cart uses Alpine `/bin/sh` with its built-in `printf`. Both require exact string equality.
- Only the configured target is recreated, with `--no-deps --force-recreate`. The collector is never recreated by the runner.
- Existing run directories and validation results are never overwritten.
- A failed validation writes `validation.json` with `research_valid: false`, then stops the entire experiment.
- If anything fails while the fault is active, the runner attempts to restore and verify the recovery value before exiting.
- The runner never uses `--remove-orphans`, changes the workload, or restarts the full stack.

## Dry run

From the repository root:

```powershell
python experiments/run_experiment.py --config experiments/configs/payment_latency_500.yaml --repetitions 5 --dry-run
```

Dry-run mode parses and validates the configuration, determines the required unused sequences without reserving them, and prints the exact command templates. It does not contact Docker or Prometheus, sleep, recreate containers, or collect telemetry.

Sequence allocation scans both fault and recovery directories and selects the lowest unused three-digit sequence. A pair is named `payment_latency_500_NNN` and `payment_latency_500_recovery_NNN`. This preserves earlier runs even when their historical fault/recovery suffixes differ.

## Real run (requires explicit operator approval)

```powershell
python experiments/run_experiment.py --config experiments/configs/payment_latency_500.yaml --repetitions 5
```

The runner checks Prometheus readiness, freshness of `traces_span_metrics_calls_total`, and all expected Compose services before activating a fault. There is no separate stabilization sleep: immediately after recreation and exact environment verification, the existing collector's configured `--warmup 30` provides the sole telemetry warm-up. This matches the timing policy used by the validated reference datasets and avoids an unintended 60-second delay.

## Standalone validation

Validation is read-only unless `--write-result` is supplied:

```powershell
python experiments/validate_run.py --config experiments/configs/payment_latency_500.yaml --run-dir rootlens/data/raw/payment_latency_500_001 --condition fault

python experiments/validate_run.py --config experiments/configs/payment_latency_500.yaml --run-dir rootlens/data/raw/payment_latency_500_recovery_002 --condition recovery --comparison-run-dir rootlens/data/raw/payment_latency_500_001
```

The gates cover configured window and row counts, service cardinality per window, exact service membership, request-rate/CPU/memory missingness, target activity, negative metrics, error-rate bounds, intervention visibility, recovery direction, and collector metadata consistency. `--write-result` refuses to replace an existing `validation.json`.

Machine-readable experiment summaries are written under `experiments/summaries/`. Run-level validity is stored beside the immutable CSV and metadata in `validation.json`.

## Cart latency experiment

Cart's healthy median latency was 12.287 ms across 360 windows from six healthy/recovery datasets; its pooled p90 was 32.692 ms. The configured 400 ms fault threshold is well outside healthy variation while allowing measurement noise around the deterministic 500 ms delay. The validator compares faults with the preserved `healthy_pilot_001` baseline and requires an increase of at least 350 ms; recovery must be no more than 25% of fault latency.

Preview the single 60-second smoke pair with:

```powershell
python experiments/run_experiment.py --config experiments/configs/cart_latency_500.yaml --smoke --dry-run
```

Removing `--dry-run` is an infrastructure-changing action and must only be done after operator approval. Smoke mode always runs exactly one pair and uses `cart_latency_smoke_NNN` / `cart_latency_smoke_recovery_NNN`; it never consumes official `cart_latency_500_NNN` sequences. Official collection retains 300-second duration, five-second sampling, a 30-second PromQL window, and the collector's sole 30-second warm-up.
