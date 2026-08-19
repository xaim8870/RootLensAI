#!/usr/bin/env python3
"""
Build PyTorch Geometric graph datasets for RootLens Dataset v1.

Inputs:
    rootlens/config/service_graph_v1.yaml

    rootlens/data/processed/dataset_v1/splits/
        train.csv
        validation.csv
        test.csv

Outputs:
    rootlens/data/processed/dataset_v1/graphs/
        train_graphs.pt
        validation_graphs.pt
        test_graphs.pt
        graph_dataset_metadata.json
        graph_preview.json

Each telemetry window becomes ONE graph:

    nodes:
        12 frozen RootLens services

    node features:
        7 telemetry metrics per service

    x shape:
        [12, 7]

    topology:
        frozen service dependency graph

    target:
        graph-level root_cause_service classification

    classes:
        0 = healthy
        1 = payment
        2 = cart
        3 = checkout
        4 = product_catalog

Graph construction policy:
    - Preserve the frozen caller -> callee topology.
    - Add reverse edges for bidirectional message passing.
    - Add explicit self-loops.
    - Deduplicate edge pairs.
    - Preserve deterministic node order from service_graph_v1.yaml.

This script DOES NOT train a GNN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch_geometric.data import Data


EXPECTED_NODE_COUNT = 12
EXPECTED_NODE_FEATURE_COUNT = 7
EXPECTED_FLAT_FEATURE_COUNT = 84

EXPECTED_SPLIT_ROWS = {
    "train": 1080,
    "validation": 360,
    "test": 360,
}

CLASS_TO_INDEX = {
    "healthy": 0,
    "payment": 1,
    "cart": 2,
    "checkout": 3,
    "product_catalog": 4,
}

INDEX_TO_CLASS = {
    value: key
    for key, value in CLASS_TO_INDEX.items()
}

NON_FEATURE_COLUMNS = {
    "run_id",
    "split",
    "run_role",
    "pair_id",
    "condition",
    "root_cause_service",
    "fault_type",
    "fault_family",
    "is_fault",
    "timestamp",
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"YAML document is not a mapping: {path}"
        )

    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_path(
    path: Path,
    repo_root: Path,
) -> Path:
    if not path.is_absolute():
        path = repo_root / path

    return path.resolve()


# ---------------------------------------------------------------------------
# Graph configuration
# ---------------------------------------------------------------------------

def validate_graph_config(
    config: dict[str, Any],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:

    node_order = list(
        config.get("node_order", [])
    )

    node_features = list(
        config.get("node_features", [])
    )

    edges = list(
        config.get("edges", [])
    )

    if len(node_order) != EXPECTED_NODE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_NODE_COUNT} nodes, "
            f"found {len(node_order)}"
        )

    if len(set(node_order)) != len(node_order):
        raise ValueError(
            "Duplicate node names found in node_order"
        )

    if len(node_features) != EXPECTED_NODE_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_NODE_FEATURE_COUNT} node features, "
            f"found {len(node_features)}"
        )

    if len(set(node_features)) != len(node_features):
        raise ValueError(
            "Duplicate node feature names found"
        )

    node_set = set(node_order)

    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise ValueError(
                f"Edge {i} is not a mapping"
            )

        source = edge.get("source")
        target = edge.get("target")

        if source not in node_set:
            raise ValueError(
                f"Edge {i}: unknown source node '{source}'"
            )

        if target not in node_set:
            raise ValueError(
                f"Edge {i}: unknown target node '{target}'"
            )

        if source == target:
            raise ValueError(
                f"Frozen call graph edge {i} is already a self-loop: "
                f"{source} -> {target}"
            )

    expected = config.get(
        "expected_counts",
        {},
    )

    expected_nodes = expected.get("nodes")
    expected_edges = expected.get(
        "directed_call_edges"
    )
    expected_features = expected.get(
        "node_features"
    )

    if expected_nodes is not None and expected_nodes != len(node_order):
        raise ValueError(
            "Graph expected_counts.nodes mismatch"
        )

    if expected_edges is not None and expected_edges != len(edges):
        raise ValueError(
            "Graph expected_counts.directed_call_edges mismatch"
        )

    if expected_features is not None and expected_features != len(node_features):
        raise ValueError(
            "Graph expected_counts.node_features mismatch"
        )

    return node_order, node_features, edges


def build_edge_index(
    node_order: list[str],
    edges: list[dict[str, Any]],
    bidirectional: bool,
    add_self_loops: bool,
) -> tuple[torch.Tensor, list[tuple[str, str]]]:
    """
    Build deterministic PyG edge_index.

    edge_index shape:
        [2, num_edges]
    """

    node_to_index = {
        service: idx
        for idx, service in enumerate(node_order)
    }

    edge_pairs: set[tuple[int, int]] = set()

    for edge in edges:
        source = node_to_index[
            edge["source"]
        ]
        target = node_to_index[
            edge["target"]
        ]

        edge_pairs.add(
            (source, target)
        )

        if bidirectional:
            edge_pairs.add(
                (target, source)
            )

    if add_self_loops:
        for idx in range(
            len(node_order)
        ):
            edge_pairs.add(
                (idx, idx)
            )

    # Stable deterministic ordering.
    ordered_pairs = sorted(
        edge_pairs
    )

    if not ordered_pairs:
        raise ValueError(
            "No graph edges generated"
        )

    edge_index = torch.tensor(
        ordered_pairs,
        dtype=torch.long,
    ).t().contiguous()

    readable_edges = [
        (
            node_order[source],
            node_order[target],
        )
        for source, target in ordered_pairs
    ]

    return edge_index, readable_edges


# ---------------------------------------------------------------------------
# Processed split validation
# ---------------------------------------------------------------------------

def get_expected_flat_feature_columns(
    node_order: list[str],
    node_features: list[str],
) -> list[str]:
    """
    This must match build_processed_dataset_v1.py:
        <service>__<metric>
    """
    return [
        f"{service}__{metric}"
        for metric in node_features
        for service in node_order
    ]


def validate_split_dataframe(
    df: pd.DataFrame,
    split_name: str,
    node_order: list[str],
    node_features: list[str],
) -> list[str]:

    expected_rows = EXPECTED_SPLIT_ROWS[
        split_name
    ]

    if len(df) != expected_rows:
        raise ValueError(
            f"{split_name}: expected {expected_rows} rows, "
            f"found {len(df)}"
        )

    required_metadata = {
        "run_id",
        "split",
        "root_cause_service",
        "timestamp",
        "fault_type",
        "fault_family",
    }

    missing_metadata = (
        required_metadata
        - set(df.columns)
    )

    if missing_metadata:
        raise ValueError(
            f"{split_name}: missing metadata columns: "
            f"{sorted(missing_metadata)}"
        )

    if not df["split"].eq(
        split_name
    ).all():
        observed = sorted(
            df["split"]
            .astype(str)
            .unique()
        )

        raise ValueError(
            f"{split_name}: split column contains "
            f"unexpected values: {observed}"
        )

    flat_features = [
        col
        for col in df.columns
        if col not in NON_FEATURE_COLUMNS
    ]

    if len(flat_features) != EXPECTED_FLAT_FEATURE_COUNT:
        raise ValueError(
            f"{split_name}: expected "
            f"{EXPECTED_FLAT_FEATURE_COUNT} flat features, "
            f"found {len(flat_features)}"
        )

    expected_features = (
        get_expected_flat_feature_columns(
            node_order=node_order,
            node_features=node_features,
        )
    )

    missing_features = sorted(
        set(expected_features)
        - set(df.columns)
    )

    extra_features = sorted(
        set(flat_features)
        - set(expected_features)
    )

    if missing_features:
        raise ValueError(
            f"{split_name}: missing expected graph features: "
            f"{missing_features}"
        )

    if extra_features:
        raise ValueError(
            f"{split_name}: unexpected graph features: "
            f"{extra_features}"
        )

    if df[
        expected_features
    ].isna().any().any():
        raise ValueError(
            f"{split_name}: graph feature matrix contains NaN values"
        )

    observed_classes = set(
        df["root_cause_service"]
        .astype(str)
        .unique()
    )

    expected_classes = set(
        CLASS_TO_INDEX
    )

    if observed_classes != expected_classes:
        raise ValueError(
            f"{split_name}: label set mismatch. "
            f"Expected={sorted(expected_classes)}, "
            f"Observed={sorted(observed_classes)}"
        )

    return expected_features


# ---------------------------------------------------------------------------
# Row -> graph conversion
# ---------------------------------------------------------------------------

def build_node_feature_tensor(
    row: pd.Series,
    node_order: list[str],
    node_features: list[str],
) -> torch.Tensor:
    """
    Convert one 84-feature flat row into:

        x.shape == [12, 7]

    Rows are services.
    Columns are telemetry features.
    """

    matrix: list[list[float]] = []

    for service in node_order:
        node_values: list[float] = []

        for metric in node_features:
            column = (
                f"{service}__{metric}"
            )

            value = row[column]

            if pd.isna(value):
                raise ValueError(
                    f"NaN encountered in "
                    f"{column}"
                )

            node_values.append(
                float(value)
            )

        matrix.append(
            node_values
        )

    x = torch.tensor(
        matrix,
        dtype=torch.float32,
    )

    expected_shape = (
        EXPECTED_NODE_COUNT,
        EXPECTED_NODE_FEATURE_COUNT,
    )

    if tuple(x.shape) != expected_shape:
        raise ValueError(
            f"Node feature shape mismatch. "
            f"Expected={expected_shape}, "
            f"Observed={tuple(x.shape)}"
        )

    return x


def row_to_graph(
    row: pd.Series,
    edge_index: torch.Tensor,
    node_order: list[str],
    node_features: list[str],
) -> Data:

    label_name = str(
        row["root_cause_service"]
    )

    if label_name not in CLASS_TO_INDEX:
        raise ValueError(
            f"Unknown target class: {label_name}"
        )

    x = build_node_feature_tensor(
        row=row,
        node_order=node_order,
        node_features=node_features,
    )

    y = torch.tensor(
        [CLASS_TO_INDEX[label_name]],
        dtype=torch.long,
    )

    graph = Data(
        x=x,
        edge_index=edge_index.clone(),
        y=y,
    )

    # PyG permits arbitrary Python metadata attributes.
    graph.run_id = str(
        row["run_id"]
    )
    graph.timestamp = str(
        row["timestamp"]
    )
    graph.split = str(
        row["split"]
    )
    graph.label_name = label_name
    graph.fault_type = str(
        row["fault_type"]
    )
    graph.fault_family = str(
        row["fault_family"]
    )

    return graph


def build_graphs_for_split(
    df: pd.DataFrame,
    edge_index: torch.Tensor,
    node_order: list[str],
    node_features: list[str],
) -> list[Data]:

    graphs: list[Data] = []

    for _, row in df.iterrows():
        graph = row_to_graph(
            row=row,
            edge_index=edge_index,
            node_order=node_order,
            node_features=node_features,
        )

        graphs.append(
            graph
        )

    return graphs


# ---------------------------------------------------------------------------
# Graph audits
# ---------------------------------------------------------------------------

def audit_graph_list(
    graphs: list[Data],
    split_name: str,
    edge_index: torch.Tensor,
) -> dict[str, Any]:

    expected_count = EXPECTED_SPLIT_ROWS[
        split_name
    ]

    if len(graphs) != expected_count:
        raise ValueError(
            f"{split_name}: expected {expected_count} graphs, "
            f"found {len(graphs)}"
        )

    class_counts = {
        class_name: 0
        for class_name in CLASS_TO_INDEX
    }

    run_counts: dict[str, int] = {}

    for idx, graph in enumerate(
        graphs
    ):
        if tuple(
            graph.x.shape
        ) != (
            EXPECTED_NODE_COUNT,
            EXPECTED_NODE_FEATURE_COUNT,
        ):
            raise ValueError(
                f"{split_name} graph {idx}: "
                f"x shape is {tuple(graph.x.shape)}"
            )

        if graph.edge_index.shape != edge_index.shape:
            raise ValueError(
                f"{split_name} graph {idx}: "
                "edge_index shape mismatch"
            )

        if not torch.equal(
            graph.edge_index,
            edge_index,
        ):
            raise ValueError(
                f"{split_name} graph {idx}: "
                "edge_index differs from frozen graph"
            )

        if torch.isnan(
            graph.x
        ).any():
            raise ValueError(
                f"{split_name} graph {idx}: x contains NaN"
            )

        label_idx = int(
            graph.y.item()
        )

        if label_idx not in INDEX_TO_CLASS:
            raise ValueError(
                f"{split_name} graph {idx}: "
                f"invalid label index {label_idx}"
            )

        expected_name = INDEX_TO_CLASS[
            label_idx
        ]

        if graph.label_name != expected_name:
            raise ValueError(
                f"{split_name} graph {idx}: "
                f"label metadata mismatch"
            )

        class_counts[
            graph.label_name
        ] += 1

        run_counts[
            graph.run_id
        ] = (
            run_counts.get(
                graph.run_id,
                0,
            )
            + 1
        )

    bad_run_counts = {
        run_id: count
        for run_id, count
        in run_counts.items()
        if count != 60
    }

    if bad_run_counts:
        raise ValueError(
            f"{split_name}: some runs do not contain "
            f"60 graphs: {bad_run_counts}"
        )

    return {
        "graphs": len(graphs),
        "runs": len(run_counts),
        "class_counts": class_counts,
        "node_feature_shape": [
            EXPECTED_NODE_COUNT,
            EXPECTED_NODE_FEATURE_COUNT,
        ],
        "edge_count": int(
            edge_index.shape[1]
        ),
    }


def audit_run_leakage(
    split_graphs: dict[str, list[Data]],
) -> dict[str, list[str]]:

    run_sets = {
        split: {
            graph.run_id
            for graph in graphs
        }
        for split, graphs
        in split_graphs.items()
    }

    intersections = {
        "train_validation": sorted(
            run_sets["train"]
            & run_sets["validation"]
        ),
        "train_test": sorted(
            run_sets["train"]
            & run_sets["test"]
        ),
        "validation_test": sorted(
            run_sets["validation"]
            & run_sets["test"]
        ),
    }

    if any(
        intersections.values()
    ):
        raise ValueError(
            f"RUN LEAKAGE DETECTED in graph dataset: "
            f"{intersections}"
        )

    return intersections


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_graph_list(
    graphs: list[Data],
    path: Path,
) -> None:
    """
    Store graph list as a torch artifact.

    Explicitly saving a Python list of PyG Data objects keeps loading simple:

        graphs = torch.load(path, weights_only=False)
    """
    torch.save(
        graphs,
        path,
    )


def make_preview(
    graph: Data,
    node_order: list[str],
    node_features: list[str],
    readable_edges: list[tuple[str, str]],
) -> dict[str, Any]:

    node_rows = []

    for node_idx, service in enumerate(
        node_order
    ):
        values = graph.x[
            node_idx
        ].tolist()

        node_rows.append(
            {
                "node_index": node_idx,
                "service": service,
                "features": {
                    metric: float(value)
                    for metric, value
                    in zip(
                        node_features,
                        values,
                    )
                },
            }
        )

    return {
        "run_id": graph.run_id,
        "timestamp": graph.timestamp,
        "split": graph.split,
        "label_name": graph.label_name,
        "label_index": int(
            graph.y.item()
        ),
        "x_shape": list(
            graph.x.shape
        ),
        "edge_index_shape": list(
            graph.edge_index.shape
        ),
        "nodes": node_rows,
        "message_passing_edges": [
            {
                "source": source,
                "target": target,
            }
            for source, target
            in readable_edges
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build PyTorch Geometric RootLens Dataset v1 graphs."
        )
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="RootLens repository root.",
    )

    parser.add_argument(
        "--graph-config",
        type=Path,
        default=Path(
            "rootlens/config/service_graph_v1.yaml"
        ),
        help="Frozen service graph YAML.",
    )

    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path(
            "rootlens/data/processed/dataset_v1/splits"
        ),
        help="Directory containing train/validation/test CSVs.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "rootlens/data/processed/dataset_v1/graphs"
        ),
        help="Graph dataset output directory.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = args.repo_root.resolve()

    graph_config_path = resolve_path(
        args.graph_config,
        repo_root,
    )

    split_dir = resolve_path(
        args.split_dir,
        repo_root,
    )

    output_dir = resolve_path(
        args.output_dir,
        repo_root,
    )

    if not graph_config_path.is_file():
        print(
            f"ERROR: graph config not found: "
            f"{graph_config_path}",
            file=sys.stderr,
        )
        return 1

    if not split_dir.is_dir():
        print(
            f"ERROR: split directory not found: "
            f"{split_dir}",
            file=sys.stderr,
        )
        return 1

    try:
        # --------------------------------------------------------------
        # Load graph definition
        # --------------------------------------------------------------

        config = load_yaml(
            graph_config_path
        )

        (
            node_order,
            node_features,
            frozen_edges,
        ) = validate_graph_config(
            config
        )

        policy = config.get(
            "gnn_policy",
            {},
        )

        bidirectional = bool(
            policy.get(
                "add_reverse_edges_at_build_time",
                True,
            )
        )

        add_self_loops = bool(
            policy.get(
                "add_self_loops",
                True,
            )
        )

        (
            edge_index,
            readable_edges,
        ) = build_edge_index(
            node_order=node_order,
            edges=frozen_edges,
            bidirectional=bidirectional,
            add_self_loops=add_self_loops,
        )

        # --------------------------------------------------------------
        # Load and convert each split
        # --------------------------------------------------------------

        split_graphs: dict[
            str,
            list[Data],
        ] = {}

        split_audits: dict[
            str,
            dict[str, Any],
        ] = {}

        source_csv_sha256: dict[
            str,
            str,
        ] = {}

        for split_name, filename in [
            ("train", "train.csv"),
            (
                "validation",
                "validation.csv",
            ),
            ("test", "test.csv"),
        ]:
            csv_path = (
                split_dir
                / filename
            )

            if not csv_path.is_file():
                raise FileNotFoundError(
                    f"Missing split CSV: "
                    f"{csv_path}"
                )

            df = pd.read_csv(
                csv_path
            )

            validate_split_dataframe(
                df=df,
                split_name=split_name,
                node_order=node_order,
                node_features=node_features,
            )

            graphs = build_graphs_for_split(
                df=df,
                edge_index=edge_index,
                node_order=node_order,
                node_features=node_features,
            )

            split_graphs[
                split_name
            ] = graphs

            split_audits[
                split_name
            ] = audit_graph_list(
                graphs=graphs,
                split_name=split_name,
                edge_index=edge_index,
            )

            source_csv_sha256[
                split_name
            ] = sha256_file(
                csv_path
            )

        # --------------------------------------------------------------
        # Leakage audit
        # --------------------------------------------------------------

        intersections = audit_run_leakage(
            split_graphs
        )

        # --------------------------------------------------------------
        # Save
        # --------------------------------------------------------------

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_paths = {}

        for split_name, graphs in (
            split_graphs.items()
        ):
            path = (
                output_dir
                / f"{split_name}_graphs.pt"
            )

            save_graph_list(
                graphs=graphs,
                path=path,
            )

            output_paths[
                split_name
            ] = path

        preview = make_preview(
            graph=split_graphs[
                "train"
            ][0],
            node_order=node_order,
            node_features=node_features,
            readable_edges=readable_edges,
        )

        preview_path = (
            output_dir
            / "graph_preview.json"
        )

        with preview_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                preview,
                f,
                indent=2,
            )

        metadata = {
            "graph_dataset_version": "v1",
            "source_graph_name": (
                config.get(
                    "graph",
                    {},
                ).get(
                    "name"
                )
            ),
            "source_graph_version": (
                config.get(
                    "graph",
                    {},
                ).get(
                    "version"
                )
            ),
            "service_graph_sha256": (
                sha256_file(
                    graph_config_path
                )
            ),
            "graph_task": (
                "graph_level_root_cause_service_classification"
            ),
            "node_order": node_order,
            "node_features": node_features,
            "class_to_index": CLASS_TO_INDEX,
            "index_to_class": {
                str(k): v
                for k, v
                in INDEX_TO_CLASS.items()
            },
            "node_count": len(
                node_order
            ),
            "node_feature_count": len(
                node_features
            ),
            "directed_call_edges": len(
                frozen_edges
            ),
            "message_passing_edge_count": int(
                edge_index.shape[1]
            ),
            "message_passing_policy": {
                "bidirectional": (
                    bidirectional
                ),
                "self_loops": (
                    add_self_loops
                ),
            },
            "edge_index": (
                edge_index.tolist()
            ),
            "split_audits": (
                split_audits
            ),
            "run_intersections": (
                intersections
            ),
            "leakage_detected": False,
            "source_split_csv_sha256": (
                source_csv_sha256
            ),
            "artifacts": {
                "train_graphs": str(
                    output_paths[
                        "train"
                    ]
                ),
                "validation_graphs": str(
                    output_paths[
                        "validation"
                    ]
                ),
                "test_graphs": str(
                    output_paths[
                        "test"
                    ]
                ),
                "preview": str(
                    preview_path
                ),
            },
            "loading_note": (
                "Load .pt graph lists with "
                "torch.load(path, weights_only=False)."
            ),
        }

        # Add graph artifact checksums.
        metadata[
            "graph_artifact_sha256"
        ] = {
            split: sha256_file(
                path
            )
            for split, path
            in output_paths.items()
        }

        metadata_path = (
            output_dir
            / "graph_dataset_metadata.json"
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metadata,
                f,
                indent=2,
            )

        # --------------------------------------------------------------
        # Console report
        # --------------------------------------------------------------

        print()
        print("=" * 86)
        print(
            "RootLens Graph Dataset v1 Build"
        )
        print("=" * 86)

        print(
            f"Graph config:           "
            f"{graph_config_path}"
        )
        print(
            f"Nodes / graph:          "
            f"{len(node_order)}"
        )
        print(
            f"Features / node:        "
            f"{len(node_features)}"
        )
        print(
            f"x tensor shape:         "
            f"[{len(node_order)}, "
            f"{len(node_features)}]"
        )
        print(
            f"Frozen call edges:      "
            f"{len(frozen_edges)}"
        )
        print(
            f"Message-passing edges:  "
            f"{edge_index.shape[1]}"
        )
        print(
            f"Bidirectional:          "
            f"{'YES' if bidirectional else 'NO'}"
        )
        print(
            f"Self-loops:             "
            f"{'YES' if add_self_loops else 'NO'}"
        )

        print()
        print("GRAPH COUNTS")
        print("-" * 86)

        for split_name in [
            "train",
            "validation",
            "test",
        ]:
            audit = split_audits[
                split_name
            ]

            print(
                f"{split_name:<18} "
                f"{audit['graphs']} graphs / "
                f"{audit['runs']} runs"
            )

        print()
        print("CLASS COUNTS")
        print("-" * 86)

        for split_name in [
            "train",
            "validation",
            "test",
        ]:
            print(
                f"{split_name}:"
            )

            counts = split_audits[
                split_name
            ][
                "class_counts"
            ]

            for class_name in (
                CLASS_TO_INDEX
            ):
                print(
                    f"  {class_name:<22} "
                    f"{counts[class_name]}"
                )

        print()
        print("LEAKAGE AUDIT")
        print("-" * 86)

        print(
            f"Train ∩ Validation:     "
            f"{len(intersections['train_validation'])}"
        )
        print(
            f"Train ∩ Test:           "
            f"{len(intersections['train_test'])}"
        )
        print(
            f"Validation ∩ Test:      "
            f"{len(intersections['validation_test'])}"
        )
        print(
            "Leakage detected:       NO"
        )

        print()
        print("OUTPUTS")
        print("-" * 86)

        print(
            f"Train graphs:      "
            f"{output_paths['train']}"
        )
        print(
            f"Validation graphs: "
            f"{output_paths['validation']}"
        )
        print(
            f"Test graphs:       "
            f"{output_paths['test']}"
        )
        print(
            f"Metadata:          "
            f"{metadata_path}"
        )
        print(
            f"Preview:           "
            f"{preview_path}"
        )

        print()
        print(
            "GRAPH DATASET STATUS: PASS"
        )
        print("=" * 86)
        print()

        return 0

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())