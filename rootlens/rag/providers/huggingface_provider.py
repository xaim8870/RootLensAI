"""Hugging Face Inference Providers chat-completion adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import InferenceClient
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[3]


DEFAULT_MODEL = "Qwen/Qwen3-4B-Thinking-2507"


class GenerationUnavailableError(RuntimeError):
    pass


SYSTEM_PROMPT = """You are the explanation layer for RootLensAI. GraphSAGE is the RCA decision engine. Never override or replace its root-cause prediction. Use only supplied evidence. Never invent metrics, logs, traces, dependencies, recovery observations, databases, queues, gateways, hosts, or external systems. Historical development incidents are supporting analogies only, not proof. Refer to the diagnosis as the model-indicated or most likely root cause according to RootLens. Every evidence-based claim must cite supplied evidence IDs. If confidence is low or probabilities compete, state uncertainty. Investigation steps are checks, never guaranteed remediation, and may name only services, telemetry features, or topology relationships explicitly present in the supplied evidence. Return only valid JSON with keys summary, evidence, uncertainty, investigate_next. evidence must be a list of objects containing claim and evidence_ids. investigate_next must contain at most three strings."""


def generate_json(
    evidence_context: str, timeout: float = 45.0
) -> tuple[dict[str, Any], dict[str, str]]:
    load_dotenv(REPO_ROOT / ".env", override=False)
    token = os.getenv("HF_TOKEN")
    if not token:
        raise GenerationUnavailableError("HF_TOKEN is not configured")
    model = os.getenv("ROOTLENS_HF_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    try:
        client = InferenceClient(provider="auto", api_key=token, timeout=timeout)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Explain this RootLens result using only these evidence items:\n" + evidence_context},
            ],
            max_tokens=700, temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        result = json.loads(content)
    except Exception as exc:
        message = str(exc).replace(token, "[REDACTED]")
        raise GenerationUnavailableError(f"Hugging Face generation unavailable: {message}") from exc
    required = {"summary", "evidence", "uncertainty", "investigate_next"}
    if not isinstance(result, dict) or not required.issubset(result):
        raise GenerationUnavailableError("Generation response did not match the required schema")
    allowed_ids = {item["id"] for item in json.loads(evidence_context)}
    cited_ids = {
        evidence_id
        for claim in result["evidence"]
        for evidence_id in claim.get("evidence_ids", [])
    }
    if not cited_ids.issubset(allowed_ids):
        raise GenerationUnavailableError("Generation cited an evidence ID that was not supplied")
    return result, {"model": model, "provider": "Hugging Face Inference Providers (auto)"}
