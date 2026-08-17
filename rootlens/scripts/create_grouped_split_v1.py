#!/usr/bin/env python3
"""
Create a leakage-safe grouped train/validation/test split for RootLens Dataset v1.

Input:
    rootlens/data/processed/dataset_v1/processed_dataset_v1.csv

Output:
    rootlens/data/processed/dataset_v1/splits/
        split_definition_v1.csv
        train.csv
        validation.csv
        test.csv
        split_metadata.json

Split policy:
    - Split at RUN level only.
    - Never split individual telemetry windows randomly.
    - Fixed seed for reproducibility.
    - 60/20/20 by runs:
        train = 18 runs
        val   = 6 runs
        test  = 6 runs
    - Preserve all 5 RCA classes in every split.
    - Preserve both payment fault families in every split.

Expected processed dataset:
    30 runs
    60 windows/run
    1800 rows total

Expected split sizes:
    train: 18 runs / 1080 rows
    val:    6 runs /  360 rows
    test:   6 runs /  360 rows
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SEED = 42

EXPECTED_TOTAL_RUNS = 30
EXPECTED_WINDOWS_PER_RUN = 60

EXPECTED_SPLIT_RUNS = {
    "train": 18,
    "validation": 6,
    "test": 6,
}

EXPECTED_SPLIT_ROWS = {
    "train": 1080,
    "validation": 360,
    "test": 360,
}

EXPECTED_CLASSES = {
    "healthy",
    "payment",
    "cart",
    "checkout",
    "product_catalog",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_processed_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {path}")

    df = pd.read_csv(path)

    required = {
        "run_id",
        "run_role",
        "root_cause_service",
        "fault_family",
        "fault_type",
        "timestamp",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Processed dataset missing required columns: {sorted(missing)}"
        )

    return df


def build_run_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the processed window-level dataset into one row per experimental run.
    """
    counts = df["run_id"].value_counts()

    bad_counts = counts[counts != EXPECTED_WINDOWS_PER_RUN]
    if not bad_counts.empty:
        raise ValueError(
            "Some runs do not contain exactly "
            f"{EXPECTED_WINDOWS_PER_RUN} windows: {bad_counts.to_dict()}"
        )

    run_table = (
        df.groupby("run_id", as_index=False)
        .agg(
            run_role=("run_role", "first"),
            root_cause_service=("root_cause_service", "first"),
            fault_family=("fault_family", "first"),
            fault_type=("fault_type", "first"),
            window_count=("timestamp", "count"),
        )
    )

    # Verify each run has consistent labels/provenance.
    consistency_columns = [
        "run_role",
        "root_cause_service",
        "fault_family",
        "fault_type",
    ]

    for col in consistency_columns:
        nunique = df.groupby("run_id")[col].nunique(dropna=False)
        inconsistent = nunique[nunique != 1]
        if not inconsistent.empty:
            raise ValueError(
                f"Column '{col}' is inconsistent within runs: "
                f"{inconsistent.to_dict()}"
            )

    if len(run_table) != EXPECTED_TOTAL_RUNS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_RUNS} unique runs, found {len(run_table)}"
        )

    observed_classes = set(run_table["root_cause_service"].astype(str).unique())
    if observed_classes != EXPECTED_CLASSES:
        raise ValueError(
            "Unexpected RCA classes. "
            f"Expected={sorted(EXPECTED_CLASSES)}, "
            f"Observed={sorted(observed_classes)}"
        )

    return run_table


def split_group(
    run_ids: list[str],
    train_n: int,
    val_n: int,
    test_n: int,
    rng: random.Random,
) -> dict[str, list[str]]:
    """
    Deterministically shuffle one homogeneous run group and allocate exact counts.
    """
    if len(run_ids) != train_n + val_n + test_n:
        raise ValueError(
            f"Group size {len(run_ids)} does not match requested "
            f"{train_n}+{val_n}+{test_n}"
        )

    ids = list(run_ids)
    rng.shuffle(ids)

    train = ids[:train_n]
    validation = ids[train_n:train_n + val_n]
    test = ids[train_n + val_n:]

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


def create_split_definition(run_table: pd.DataFrame) -> pd.DataFrame:
    """
    Build exact run-level split assignments.

    Strategy:
    - healthy:            3 / 1 / 1
    - cart latency:       3 / 1 / 1
    - checkout latency:   3 / 1 / 1
    - product error:      3 / 1 / 1
    - payment latency:    3 / 1 / 1
    - payment error:      3 / 1 / 1

    This yields:
    - train: 18 runs
    - val:    6 runs
    - test:   6 runs
    """
    rng = random.Random(SEED)

    grouped_assignments: list[dict[str, Any]] = []

    # Define strata explicitly so payment fault families remain balanced.
    strata = [
        {
            "name": "healthy",
            "mask": (
                run_table["root_cause_service"].eq("healthy")
            ),
        },
        {
            "name": "cart_latency",
            "mask": (
                run_table["root_cause_service"].eq("cart")
                & run_table["fault_type"].eq("latency")
            ),
        },
        {
            "name": "checkout_latency",
            "mask": (
                run_table["root_cause_service"].eq("checkout")
                & run_table["fault_type"].eq("latency")
            ),
        },
        {
            "name": "product_catalog_error",
            "mask": (
                run_table["root_cause_service"].eq("product_catalog")
                & run_table["fault_type"].eq("error")
            ),
        },
        {
            "name": "payment_latency",
            "mask": (
                run_table["root_cause_service"].eq("payment")
                & run_table["fault_type"].eq("latency")
            ),
        },
        {
            "name": "payment_error",
            "mask": (
                run_table["root_cause_service"].eq("payment")
                & run_table["fault_type"].eq("error")
            ),
        },
    ]

    used_runs: set[str] = set()

    for stratum in strata:
        subset = run_table.loc[stratum["mask"]].copy()
        run_ids = subset["run_id"].tolist()

        if len(run_ids) != 5:
            raise ValueError(
                f"Stratum '{stratum['name']}' expected 5 runs, "
                f"found {len(run_ids)}: {run_ids}"
            )

        split = split_group(
            run_ids=run_ids,
            train_n=3,
            val_n=1,
            test_n=1,
            rng=rng,
        )

        for split_name, ids in split.items():
            for run_id in ids:
                if run_id in used_runs:
                    raise ValueError(
                        f"Run assigned more than once: {run_id}"
                    )
                used_runs.add(run_id)

                row = subset.loc[subset["run_id"].eq(run_id)].iloc[0]

                grouped_assignments.append(
                    {
                        "run_id": run_id,
                        "run_role": row["run_role"],
                        "root_cause_service": row["root_cause_service"],
                        "fault_family": row["fault_family"],
                        "fault_type": row["fault_type"],
                        "window_count": int(row["window_count"]),
                        "split_stratum": stratum["name"],
                        "split": split_name,
                        "split_seed": SEED,
                    }
                )

    all_runs = set(run_table["run_id"])
    if used_runs != all_runs:
        missing = sorted(all_runs - used_runs)
        extra = sorted(used_runs - all_runs)
        raise ValueError(
            f"Run assignment mismatch. Missing={missing}, Extra={extra}"
        )

    split_df = pd.DataFrame(grouped_assignments)

    # Deterministic presentation order.
    split_order = {"train": 0, "validation": 1, "test": 2}
    split_df["_split_order"] = split_df["split"].map(split_order)

    split_df = (
        split_df
        .sort_values(
            [
                "_split_order",
                "root_cause_service",
                "fault_type",
                "run_id",
            ]
        )
        .drop(columns="_split_order")
        .reset_index(drop=True)
    )

    return split_df


def audit_split(
    processed: pd.DataFrame,
    split_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Hard leakage and distribution checks.
    """
    run_sets = {
        split_name: set(
            split_df.loc[
                split_df["split"].eq(split_name),
                "run_id",
            ]
        )
        for split_name in EXPECTED_SPLIT_RUNS
    }

    # ---------------------------------------------------------------
    # Hard leakage checks
    # ---------------------------------------------------------------

    intersections = {
        "train_validation": sorted(
            run_sets["train"] & run_sets["validation"]
        ),
        "train_test": sorted(
            run_sets["train"] & run_sets["test"]
        ),
        "validation_test": sorted(
            run_sets["validation"] & run_sets["test"]
        ),
    }

    if any(intersections.values()):
        raise ValueError(
            f"RUN LEAKAGE DETECTED: {intersections}"
        )

    # ---------------------------------------------------------------
    # Run-count checks
    # ---------------------------------------------------------------

    observed_run_counts = (
        split_df["split"].value_counts().to_dict()
    )

    for split_name, expected in EXPECTED_SPLIT_RUNS.items():
        observed = int(observed_run_counts.get(split_name, 0))
        if observed != expected:
            raise ValueError(
                f"{split_name}: expected {expected} runs, found {observed}"
            )

    # ---------------------------------------------------------------
    # Class coverage checks
    # ---------------------------------------------------------------

    class_counts_by_split: dict[str, dict[str, int]] = {}

    for split_name in EXPECTED_SPLIT_RUNS:
        subset = split_df[split_df["split"].eq(split_name)]

        observed_classes = set(
            subset["root_cause_service"].astype(str).unique()
        )

        if observed_classes != EXPECTED_CLASSES:
            raise ValueError(
                f"{split_name}: class coverage mismatch. "
                f"Expected={sorted(EXPECTED_CLASSES)}, "
                f"Observed={sorted(observed_classes)}"
            )

        class_counts_by_split[split_name] = (
            subset["root_cause_service"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

    # ---------------------------------------------------------------
    # Payment family coverage checks
    # ---------------------------------------------------------------

    payment_family_counts: dict[str, dict[str, int]] = {}

    for split_name in EXPECTED_SPLIT_RUNS:
        subset = split_df[
            split_df["split"].eq(split_name)
            & split_df["root_cause_service"].eq("payment")
        ]

        families = (
            subset["fault_type"]
            .value_counts()
            .to_dict()
        )

        if families.get("latency", 0) != (
            3 if split_name == "train" else 1
        ):
            raise ValueError(
                f"{split_name}: payment latency family allocation incorrect: "
                f"{families}"
            )

        if families.get("error", 0) != (
            3 if split_name == "train" else 1
        ):
            raise ValueError(
                f"{split_name}: payment error family allocation incorrect: "
                f"{families}"
            )

        payment_family_counts[split_name] = families

    # ---------------------------------------------------------------
    # Row-count checks after materialization
    # ---------------------------------------------------------------

    row_counts_by_split: dict[str, int] = {}

    for split_name in EXPECTED_SPLIT_ROWS:
        ids = run_sets[split_name]
        subset = processed[processed["run_id"].isin(ids)]

        observed_rows = len(subset)
        expected_rows = EXPECTED_SPLIT_ROWS[split_name]

        if observed_rows != expected_rows:
            raise ValueError(
                f"{split_name}: expected {expected_rows} rows, "
                f"found {observed_rows}"
            )

        row_counts_by_split[split_name] = observed_rows

        # Each run must still contribute 60 windows.
        per_run = subset["run_id"].value_counts()
        bad = per_run[per_run != EXPECTED_WINDOWS_PER_RUN]

        if not bad.empty:
            raise ValueError(
                f"{split_name}: run window counts incorrect: {bad.to_dict()}"
            )

    return {
        "run_intersections": intersections,
        "run_counts_by_split": observed_run_counts,
        "row_counts_by_split": row_counts_by_split,
        "class_counts_by_split": class_counts_by_split,
        "payment_fault_type_counts_by_split": payment_family_counts,
        "leakage_detected": False,
    }


def materialize_split_csvs(
    processed: pd.DataFrame,
    split_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    split_map = dict(
        zip(split_df["run_id"], split_df["split"])
    )

    temp = processed.copy()
    temp["split"] = temp["run_id"].map(split_map)

    if temp["split"].isna().any():
        missing_runs = sorted(
            temp.loc[temp["split"].isna(), "run_id"].unique()
        )
        raise ValueError(
            f"Processed rows missing split assignment: {missing_runs}"
        )

    for split_name, filename in [
        ("train", "train.csv"),
        ("validation", "validation.csv"),
        ("test", "test.csv"),
    ]:
        subset = temp[temp["split"].eq(split_name)].copy()

        # Put split beside run_id for readability.
        split_col = subset.pop("split")
        subset.insert(1, "split", split_col)

        path = output_dir / filename
        subset.to_csv(path, index=False)
        paths[split_name] = path

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grouped run-level split for RootLens Dataset v1."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "rootlens/data/processed/dataset_v1/processed_dataset_v1.csv"
        ),
        help="Processed Dataset v1 CSV.",
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
        default=Path(
            "rootlens/data/processed/dataset_v1/splits"
        ),
        help="Split output directory.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = args.repo_root.resolve()

    dataset_path = args.dataset
    if not dataset_path.is_absolute():
        dataset_path = repo_root / dataset_path
    dataset_path = dataset_path.resolve()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir = output_dir.resolve()

    try:
        processed = load_processed_dataset(dataset_path)
        run_table = build_run_table(processed)
        split_df = create_split_definition(run_table)

        audit = audit_split(
            processed=processed,
            split_df=split_df,
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        split_definition_path = (
            output_dir / "split_definition_v1.csv"
        )
        split_df.to_csv(split_definition_path, index=False)

        split_paths = materialize_split_csvs(
            processed=processed,
            split_df=split_df,
            output_dir=output_dir,
        )

        metadata = {
            "split_version": "v1",
            "split_seed": SEED,
            "split_strategy": (
                "run_level_stratified_fixed_allocation"
            ),
            "grouping_key": "run_id",
            "random_row_split_allowed": False,
            "source_dataset": str(dataset_path),
            "source_dataset_sha256": sha256_file(dataset_path),
            "expected_windows_per_run": EXPECTED_WINDOWS_PER_RUN,
            "expected_total_runs": EXPECTED_TOTAL_RUNS,
            "run_counts_by_split": audit["run_counts_by_split"],
            "row_counts_by_split": audit["row_counts_by_split"],
            "class_counts_by_split": audit["class_counts_by_split"],
            "payment_fault_type_counts_by_split": (
                audit["payment_fault_type_counts_by_split"]
            ),
            "run_intersections": audit["run_intersections"],
            "leakage_detected": audit["leakage_detected"],
            "split_definition_sha256": sha256_file(
                split_definition_path
            ),
            "artifacts": {
                "split_definition": str(split_definition_path),
                "train_csv": str(split_paths["train"]),
                "validation_csv": str(split_paths["validation"]),
                "test_csv": str(split_paths["test"]),
            },
        }

        metadata_path = output_dir / "split_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print()
        print("=" * 78)
        print("RootLens Dataset v1 Grouped Split")
        print("=" * 78)
        print(f"Split seed:              {SEED}")
        print(f"Grouping key:            run_id")
        print(f"Random row split:        FORBIDDEN")
        print()

        print("RUN COUNTS")
        print("-" * 78)
        for split_name in ["train", "validation", "test"]:
            print(
                f"{split_name:<18} "
                f"{audit['run_counts_by_split'][split_name]} runs"
            )

        print()
        print("ROW COUNTS")
        print("-" * 78)
        for split_name in ["train", "validation", "test"]:
            print(
                f"{split_name:<18} "
                f"{audit['row_counts_by_split'][split_name]} rows"
            )

        print()
        print("CLASS COUNTS BY RUN")
        print("-" * 78)
        for split_name in ["train", "validation", "test"]:
            print(f"{split_name}:")
            counts = audit["class_counts_by_split"][split_name]
            for label in sorted(counts):
                print(f"  {label:<22} {counts[label]}")

        print()
        print("PAYMENT FAULT-TYPE COVERAGE")
        print("-" * 78)
        for split_name in ["train", "validation", "test"]:
            counts = audit[
                "payment_fault_type_counts_by_split"
            ][split_name]
            print(
                f"{split_name:<18} "
                f"latency={counts.get('latency', 0)}, "
                f"error={counts.get('error', 0)}"
            )

        print()
        print("LEAKAGE AUDIT")
        print("-" * 78)
        print(
            f"Train ∩ Validation:      "
            f"{len(audit['run_intersections']['train_validation'])}"
        )
        print(
            f"Train ∩ Test:            "
            f"{len(audit['run_intersections']['train_test'])}"
        )
        print(
            f"Validation ∩ Test:       "
            f"{len(audit['run_intersections']['validation_test'])}"
        )
        print(f"Leakage detected:        NO")

        print()
        print("OUTPUTS")
        print("-" * 78)
        print(f"Definition: {split_definition_path}")
        print(f"Train:      {split_paths['train']}")
        print(f"Validation: {split_paths['validation']}")
        print(f"Test:       {split_paths['test']}")
        print(f"Metadata:   {metadata_path}")
        print()
        print(f"SPLIT STATUS: PASS")
        print("=" * 78)
        print()

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())