"""Two-channel local retrieval for current RootLens RCA results."""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_ROOT = REPO_ROOT / "rootlens/rag/indexes"


class RetrievalUnavailableError(RuntimeError):
    pass


class HybridRetriever:
    def __init__(self) -> None:
        initialization_started = time.perf_counter()
        numeric_path = INDEX_ROOT / "numeric_incident_index.npz"
        metadata_path = INDEX_ROOT / "numeric_index_metadata.json"
        documents_path = INDEX_ROOT / "text_index/documents.json"
        embeddings_path = INDEX_ROOT / "text_index/embeddings.npy"
        for path in (numeric_path, metadata_path, documents_path, embeddings_path):
            if not path.is_file():
                raise RetrievalUnavailableError(f"Missing retrieval artifact: {path}")
        self.numeric = np.load(numeric_path, allow_pickle=False)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata["sealed_test_run_count_indexed"] != 0:
            raise RetrievalUnavailableError("Numeric index provenance includes sealed-test runs")
        self.features = list(self.metadata["feature_ordering"])
        self.documents = json.loads(documents_path.read_text(encoding="utf-8"))
        self.text_embeddings = np.load(embeddings_path)
        provenance = json.loads(
            (REPO_ROOT / "rootlens/rag/rag_provenance.json").read_text(encoding="utf-8")
        )
        self.encoder = SentenceTransformer(
            provenance["embedding_model"],
            cache_folder=str(INDEX_ROOT / "text_index/model_cache"),
        )
        self.neighbors = NearestNeighbors(metric="cosine").fit(self.numeric["vectors"])
        self.initialization_seconds = time.perf_counter() - initialization_started

    def _vector(self, service_metrics: dict[str, dict[str, Any]]) -> np.ndarray:
        values = []
        for feature in self.features:
            service, metric = feature.split("__", 1)
            value = service_metrics[service][metric]
            if value is None and metric in {"latency_ms", "error_rate"}:
                value = 0.0
            if value is None:
                raise RetrievalUnavailableError(f"Missing required live feature: {feature}")
            values.append(float(value))
        raw = np.asarray(values, dtype=np.float64)
        return ((raw - self.numeric["scaler_mean"]) / self.numeric["scaler_scale"]).astype(np.float32)

    def retrieve(
        self, rca_result: dict[str, Any], numeric_k: int = 3, semantic_k: int = 3
    ) -> dict[str, Any]:
        retrieval_started = time.perf_counter()
        vector = self._vector(rca_result["service_metrics"])
        numeric_started = time.perf_counter()
        index_size = len(self.numeric["vectors"])
        candidate_count = min(index_size, max(numeric_k * 10, 30))
        while True:
            distances, indices = self.neighbors.kneighbors(
                vector.reshape(1, -1), n_neighbors=candidate_count
            )
            numeric_matches = []
            seen_runs: set[str] = set()
            for distance, index in zip(distances[0], indices[0]):
                run_id = str(self.numeric["run_ids"][index])
                if run_id in seen_runs:
                    continue
                seen_runs.add(run_id)
                numeric_matches.append({
                    "run_id": run_id, "similarity": float(max(0.0, 1.0 - distance)),
                    "root_cause": str(self.numeric["root_causes"][index]),
                    "fault_type": str(self.numeric["fault_types"][index]),
                    "fault_family": str(self.numeric["fault_families"][index]),
                    "split": str(self.numeric["splits"][index]),
                    "timestamp": str(self.numeric["timestamps"][index]),
                })
                if len(numeric_matches) >= numeric_k:
                    break
            if len(numeric_matches) >= numeric_k or candidate_count == index_size:
                break
            candidate_count = min(index_size, candidate_count * 2)
        numeric_seconds = time.perf_counter() - numeric_started
        predicted = str(rca_result["predicted_root_cause"])
        target_metrics = rca_result["service_metrics"].get(predicted.replace("_", "-"), {})
        query = (
            f"RootLens model-indicated root cause {predicted}. "
            f"Probabilities {rca_result['probabilities']}. Target telemetry {target_metrics}. "
            "Relevant historical incident, topology, and protocol context."
        )
        embedding_started = time.perf_counter()
        query_embedding = self.encoder.encode([query], normalize_embeddings=True)
        embedding_seconds = time.perf_counter() - embedding_started
        similarity_started = time.perf_counter()
        scores = cosine_similarity(query_embedding, self.text_embeddings)[0]
        semantic_matches = []
        for index in np.argsort(scores)[::-1][:semantic_k]:
            document = dict(self.documents[int(index)])
            document["similarity"] = float(scores[index])
            semantic_matches.append(document)
        similarity_seconds = time.perf_counter() - similarity_started
        sealed = set(self.metadata["excluded_sealed_test_run_ids"])
        if any(match["run_id"] in sealed for match in numeric_matches):
            raise RetrievalUnavailableError("Sealed-test run appeared in numeric retrieval")
        if any(document.get("run_id") in sealed for document in semantic_matches):
            raise RetrievalUnavailableError("Sealed-test document appeared in semantic retrieval")
        return {
            "numeric_matches": numeric_matches,
            "semantic_matches": semantic_matches,
            "retrieval_timing": {
                "numeric_retrieval_seconds": numeric_seconds,
                "semantic_query_embedding_seconds": embedding_seconds,
                "semantic_similarity_seconds": similarity_seconds,
                "total_retrieval_seconds": time.perf_counter() - retrieval_started,
                "numeric_candidate_count": candidate_count,
            },
        }


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """Return the process-wide immutable historical-index retriever."""

    return HybridRetriever()
