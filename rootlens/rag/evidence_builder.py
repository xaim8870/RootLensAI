"""Concise, traceable evidence assembly for the hosted explanation model."""

from __future__ import annotations

import json
from typing import Any


def build_evidence(rca_result: dict[str, Any], retrieval: dict[str, Any]) -> dict[str, Any]:
    predicted = str(rca_result["predicted_root_cause"])
    service = predicted.replace("_", "-")
    items = [{
        "id": "E1", "kind": "current_rca",
        "content": {
            "model_indicated_root_cause": predicted,
            "confidence": rca_result["confidence"],
            "probabilities": rca_result["probabilities"],
            "target_service_metrics": rca_result["service_metrics"].get(service),
            "graph_shape": rca_result["graph_shape"],
            "promql_window": rca_result.get("promql_window", "30s"),
        },
    }]
    for match in retrieval["numeric_matches"]:
        items.append({"id": f"E{len(items) + 1}", "kind": "numeric_match", "content": match})
    for match in retrieval["semantic_matches"]:
        items.append({"id": f"E{len(items) + 1}", "kind": "semantic_context", "content": match})
    return {"evidence": items, "prompt_context": json.dumps(items, indent=2)}
