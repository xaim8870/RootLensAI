"""High-level optional RootLens hybrid-RAG explanation API."""

from __future__ import annotations

import time
import re
from datetime import datetime, timezone
from typing import Any

from rootlens.rag.evidence_builder import build_evidence
from rootlens.rag.hybrid_retriever import get_retriever
from rootlens.rag.providers.huggingface_provider import generate_json


def _safety_postprocess(explanation: dict[str, Any], rca_result: dict[str, Any]) -> dict[str, Any]:
    """Enforce analogy/uncertainty language after provider generation."""

    def soften(text: str) -> str:
        replacements = {
            r"\bconfirming\b": "supporting",
            r"\bconfirms\b": "supports",
            r"\bprove(?:s|d)?\b": "supports",
            r"\ball evidence points to\b": "retrieved evidence is consistent with",
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    result = dict(explanation)
    result["summary"] = soften(str(result["summary"]))
    result["evidence"] = [
        {**claim, "claim": soften(str(claim.get("claim", "")))}
        for claim in result["evidence"]
    ]
    uncertainty = str(result["uncertainty"])
    if uncertainty.strip().lower().startswith(("none", "no uncertainty")):
        uncertainty = (
            "Residual uncertainty remains because this is a model inference and historical "
            "development incidents are supporting analogies, not proof of the live diagnosis."
        )
    result["uncertainty"] = soften(uncertainty)
    predicted = str(rca_result["predicted_root_cause"]).replace("_", "-")
    result["investigate_next"] = [
        f"Check the current {predicted} latency, error rate, error RPS, and request rate telemetry.",
        "Compare the next 30-second telemetry window with the current class probabilities.",
        "Review only the service relationships present in the retrieved frozen topology context.",
    ]
    return result


def generate_rca_explanation(rca_result: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    cache_before = get_retriever.cache_info()
    retriever_started = time.perf_counter()
    retriever = get_retriever()
    retriever_elapsed = time.perf_counter() - retriever_started
    cache_after = get_retriever.cache_info()
    retriever_initialization_seconds = (
        retriever_elapsed if cache_after.misses > cache_before.misses else 0.0
    )
    retrieval = retriever.retrieve(rca_result)
    evidence = build_evidence(rca_result, retrieval)
    generation_started = time.perf_counter()
    explanation, provider = generate_json(evidence["prompt_context"])
    hf_generation_seconds = time.perf_counter() - generation_started
    explanation = _safety_postprocess(explanation, rca_result)
    total_explanation_seconds = time.perf_counter() - started
    retrieval_timing = retrieval["retrieval_timing"]
    return {
        "explanation": explanation,
        "retrieved_evidence": retrieval,
        "generation_metadata": {
            **provider,
            "numeric_retrieval_count": len(retrieval["numeric_matches"]),
            "semantic_retrieval_count": len(retrieval["semantic_matches"]),
            "latency_seconds": total_explanation_seconds,
            "retriever_initialization_seconds": retriever_initialization_seconds,
            "numeric_retrieval_seconds": retrieval_timing["numeric_retrieval_seconds"],
            "semantic_query_embedding_seconds": retrieval_timing["semantic_query_embedding_seconds"],
            "semantic_similarity_seconds": retrieval_timing["semantic_similarity_seconds"],
            "total_retrieval_seconds": retrieval_timing["total_retrieval_seconds"],
            "numeric_candidate_count": retrieval_timing["numeric_candidate_count"],
            "hf_generation_seconds": hf_generation_seconds,
            "total_explanation_seconds": total_explanation_seconds,
            "generated_timestamp": datetime.now(timezone.utc).isoformat(),
            "safety_postprocessed": True,
        },
    }
