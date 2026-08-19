"""Build sealed-test-safe local numeric and semantic retrieval indexes."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "rootlens/data/processed/dataset_v2"
SPLIT_ROOT = DATASET_ROOT / "splits"
INDEX_ROOT = REPO_ROOT / "rootlens/rag/indexes"
TEXT_ROOT = INDEX_ROOT / "text_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def recovery_map() -> dict[str, str]:
    manifest = yaml.safe_load(
        (REPO_ROOT / "rootlens/data/manifests/dataset_v2.yaml").read_text(encoding="utf-8")
    )
    return {
        str(pair["fault_run_id"]): str(pair["recovery_run_id"])
        for pair in manifest["accepted"]["fault_pairs"]
    }


def topology_documents() -> list[dict[str, str]]:
    topology = yaml.safe_load(
        (REPO_ROOT / "rootlens/config/service_graph_v1.yaml").read_text(encoding="utf-8")
    )
    edges = topology["edges"]
    documents = []
    for service in topology["node_order"]:
        outgoing = [edge["target"] for edge in edges if edge["source"] == service]
        incoming = [edge["source"] for edge in edges if edge["target"] == service]
        documents.append({
            "evidence_id": f"TOPOLOGY_{service}", "kind": "topology",
            "title": f"{service} topology context",
            "text": (
                f"Service topology for {service}. Calls: {', '.join(outgoing) or 'none in the frozen graph'}. "
                f"Called by: {', '.join(incoming) or 'none in the frozen graph'}. "
                "This is the frozen RootLens directed topology; retrieval context is not proof of causality."
            ),
        })
    documents.append({
        "evidence_id": "PROTOCOL_DATASET_V2", "kind": "protocol",
        "title": "Dataset v2 telemetry protocol",
        "text": (
            "RootLens Dataset v2 uses 30-second PromQL windows sampled every 5 seconds. "
            "Each system window contains 12 services and seven features per service: CPU, memory, "
            "request rate, has-requests, latency milliseconds, error RPS, and error rate. "
            "Historical development incidents are controlled analogies, not proof of a live diagnosis."
        ),
    })
    return documents


def main() -> int:
    split_path = SPLIT_ROOT / "split_definition_v2.csv"
    split_meta_path = SPLIT_ROOT / "split_metadata.json"
    processed_path = DATASET_ROOT / "processed_dataset_v2.csv"
    feature_path = DATASET_ROOT / "feature_columns.txt"
    split_meta = json.loads(split_meta_path.read_text(encoding="utf-8"))
    if sha256(split_path) != split_meta["split_definition_sha256"]:
        raise RuntimeError("Canonical split checksum mismatch; refusing to build indexes")
    split = pd.read_csv(split_path)
    split_lookup = dict(zip(split["run_id"].astype(str), split["split"].astype(str)))
    train_ids = split.loc[split["split"] == "train", "run_id"].astype(str).tolist()
    validation_ids = split.loc[split["split"] == "validation", "run_id"].astype(str).tolist()
    test_ids = split.loc[split["split"] == "test", "run_id"].astype(str).tolist()
    development_ids = train_ids + validation_ids
    if set(development_ids) & set(test_ids):
        raise RuntimeError("Sealed-test overlap detected")
    features = [line.strip() for line in feature_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(features) != 84:
        raise RuntimeError(f"Expected 84 features, found {len(features)}")
    frame = pd.read_csv(processed_path)
    development = frame[frame["run_id"].isin(development_ids)].copy()
    if set(development["run_id"]) != set(development_ids):
        raise RuntimeError("Development run population is incomplete")
    if set(development["run_id"]) & set(test_ids):
        raise RuntimeError("Sealed test entered development frame")
    if development[features].isna().any().any():
        raise RuntimeError("Model-ready development features contain NaNs")
    train_rows = development[development["run_id"].isin(train_ids)]
    scaler = StandardScaler().fit(train_rows[features].to_numpy(dtype=np.float64))
    vectors = scaler.transform(development[features].to_numpy(dtype=np.float64)).astype(np.float32)
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        INDEX_ROOT / "numeric_incident_index.npz",
        vectors=vectors, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
        run_ids=np.asarray(development["run_id"].astype(str), dtype=str),
        timestamps=np.asarray(development["timestamp"].astype(str), dtype=str),
        root_causes=np.asarray(development["root_cause_service"].astype(str), dtype=str),
        fault_types=np.asarray(development["fault_type"].astype(str), dtype=str),
        fault_families=np.asarray(development["fault_family"].astype(str), dtype=str),
        conditions=np.asarray(development["condition"].astype(str), dtype=str),
        splits=np.asarray(development["run_id"].map(split_lookup).astype(str), dtype=str),
    )
    recoveries = recovery_map()
    documents: list[dict[str, str]] = []
    for run_id, group in development.groupby("run_id", sort=True):
        row = group.iloc[0]
        root = str(row["root_cause_service"])
        target = root.replace("_", "-")
        if root == "healthy":
            summary = "No controlled fault was active; this is an accepted healthy development run."
        else:
            values = []
            for metric in ("latency_ms", "error_rate", "error_rps", "request_rate"):
                values.append(f"median {metric}={group[f'{target}__{metric}'].median():.6g}")
            summary = "; ".join(values)
        recovery = recoveries.get(str(run_id), "none for healthy run")
        documents.append({
            "evidence_id": f"INCIDENT_{run_id}", "kind": "incident", "title": str(run_id),
            "run_id": str(run_id), "root_cause": root, "fault_type": str(row["fault_type"]),
            "text": (
                f"Validated Dataset v2 development run {run_id}. Root cause label: {root}. "
                f"Fault family: {row['fault_family']}. Fault type: {row['fault_type']}. "
                f"Condition: {row['condition']}. Dataset split: {split_lookup[run_id]}. "
                f"Observed telemetry summary: {summary}. Recovery run: {recovery}. "
                "This historical run is a supporting analogy only."
            ),
        })
    documents.extend(topology_documents())
    TEXT_ROOT.mkdir(parents=True, exist_ok=True)
    documents_path = TEXT_ROOT / "documents.json"
    documents_path.write_text(json.dumps(documents, indent=2) + "\n", encoding="utf-8")
    encoder = SentenceTransformer(
        EMBEDDING_MODEL, cache_folder=str(TEXT_ROOT / "model_cache")
    )
    embeddings = encoder.encode(
        [doc["text"] for doc in documents], normalize_embeddings=True, show_progress_bar=True
    )
    np.save(TEXT_ROOT / "embeddings.npy", np.asarray(embeddings, dtype=np.float32))
    created = datetime.now(timezone.utc).isoformat()
    numeric_metadata = {
        "dataset_version": "v2", "included_splits": ["train", "validation"],
        "included_run_ids": development_ids, "excluded_sealed_test_run_ids": test_ids,
        "train_run_count": len(train_ids), "validation_run_count": len(validation_ids),
        "sealed_test_run_count_indexed": 0, "feature_ordering": features,
        "normalization_source": "canonical Dataset v2 training rows only",
        "index_algorithm": "StandardScaler + sklearn NearestNeighbors cosine; run-diverse results",
        "created_timestamp": created,
        "source_checksums": {"processed_dataset": sha256(processed_path), "split_definition": sha256(split_path)},
    }
    (INDEX_ROOT / "numeric_index_metadata.json").write_text(
        json.dumps(numeric_metadata, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "dataset_version": "v2", "canonical_split_sha256": sha256(split_path),
        "train_run_count_indexed": len(train_ids), "validation_run_count_indexed": len(validation_ids),
        "sealed_test_run_count_indexed": 0, "sealed_test_exclusion_verified": True,
        "numeric_retrieval_method": "StandardScaler(train only) + cosine nearest neighbors",
        "embedding_model": EMBEDDING_MODEL,
        "generation_model_default": "Qwen/Qwen3-4B-Thinking-2507",
        "evidence_document_count": len(documents), "creation_timestamp": created,
        "git_commit": current_git_commit(),
    }
    (REPO_ROOT / "rootlens/rag/rag_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", **provenance}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
