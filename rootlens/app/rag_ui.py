"""RAG / LLM incident analysis presentation."""

from __future__ import annotations

import html
import logging
import time

import streamlit as st

from rootlens.app.runtime_ui import (
    format_seconds,
    render_html,
    section,
)

LOGGER = logging.getLogger("rootlens.mvp")


def _safe_text(value: object) -> str:
    """Return display-safe plain text."""
    return str(value) if value is not None else ""


def _render_ai_header() -> None:
    """Render the static RAG header panel."""

    render_html(
        """
        <div class="rl-ai-shell">
          <div class="rl-ai-kicker">ROOTLENSAI · GROUNDED INCIDENT INTELLIGENCE</div>
          <div class="rl-ai-title">AI-assisted incident investigation</div>
          <div class="rl-ai-copy">
            Retrieved validated incidents + grounded explanation. GraphSAGE
            remains the independent RCA decision engine; fault-injection metadata
            is never supplied to the model or explanation pipeline.
          </div>
          <div class="rl-ai-chip-row">
            <span class="rl-chip">GRAPH RCA INDEPENDENT</span>
            <span class="rl-chip">LOCAL RETRIEVAL EVIDENCE</span>
            <span class="rl-chip">UNCERTAINTY PRESERVED</span>
          </div>
        </div>
        """
    )


def _render_report(analysis: dict) -> None:
    """Render the generated AI report with native Streamlit components.

    Native Streamlit components are used intentionally for dynamic content.
    This avoids Markdown interpreting indented/raw HTML fragments as code.
    """

    explanation = analysis.get("explanation", {})

    with st.container(border=True):
        header_left, header_right = st.columns([4, 1])

        with header_left:
            st.markdown("### AI INCIDENT REPORT")

        with header_right:
            st.caption(
                "Generated in "
                f"{format_seconds(st.session_state.get('last_rag_duration'))}"
            )

        st.markdown("#### Summary")
        st.write(_safe_text(explanation.get("summary", "")))

        st.divider()

        st.markdown("#### Supporting Evidence")

        evidence = explanation.get("evidence", [])
        if evidence:
            for index, claim in enumerate(evidence, start=1):
                with st.container(border=True):
                    left, right = st.columns([0.45, 5.55])

                    with left:
                        st.markdown(f"**E{index:02d}**")

                    with right:
                        st.write(_safe_text(claim.get("claim", "")))

                        references = ", ".join(
                            _safe_text(item)
                            for item in claim.get("evidence_ids", [])
                        )

                        if references:
                            st.caption(f"References: {references}")
        else:
            st.caption("No supporting evidence claims were returned.")

        st.divider()

        st.markdown("#### Uncertainty")
        st.write(_safe_text(explanation.get("uncertainty", "")))

        st.divider()

        st.markdown("#### Investigate Next")

        investigate_next = explanation.get("investigate_next", [])
        if investigate_next:
            for index, item in enumerate(investigate_next, start=1):
                with st.container(border=True):
                    left, right = st.columns([0.45, 5.55])

                    with left:
                        st.markdown(f"**{index:02d}**")

                    with right:
                        st.write(_safe_text(item))
        else:
            st.caption("No next investigation steps were returned.")


def _render_numeric_matches(numeric_matches: list[dict]) -> None:
    """Render similar validated incidents without raw dynamic HTML."""

    st.markdown("### Similar validated incidents")

    if not numeric_matches:
        st.caption("No numeric incident matches were returned.")
        return

    # Three cards per row. This stays responsive enough for the current UI and
    # avoids concatenating raw HTML strings.
    for start in range(0, len(numeric_matches), 3):
        row = numeric_matches[start : start + 3]
        columns = st.columns(len(row), gap="medium")

        for column, match in zip(columns, row):
            with column:
                with st.container(border=True):
                    st.markdown(
                        f"**{_safe_text(match.get('run_id', 'Unknown incident'))}**"
                    )
                    st.markdown(
                        f"`similarity {float(match.get('similarity', 0.0)):.3f}`"
                    )
                    st.caption(
                        "root cause: "
                        f"{_safe_text(match.get('root_cause', 'unknown'))}"
                    )
                    st.caption(
                        "fault: "
                        f"{_safe_text(match.get('fault_type', 'unknown'))}"
                    )
                    st.caption(
                        "split: "
                        f"{_safe_text(match.get('split', 'unknown'))}"
                    )


def _render_semantic_matches(semantic_matches: list[dict]) -> None:
    """Render retrieved knowledge/context documents."""

    st.markdown("### Knowledge / context")

    if not semantic_matches:
        st.caption("No semantic knowledge/context matches were returned.")
        return

    for document in semantic_matches:
        with st.container(border=True):
            title = _safe_text(document.get("title", "Untitled context"))
            similarity = float(document.get("similarity", 0.0))

            st.markdown(f"**{title}**")
            st.caption(f"Similarity: {similarity:.3f}")
            st.write(_safe_text(document.get("text", "")))


def _render_generation_metadata(analysis: dict) -> None:
    """Render model/provider/generation metadata."""

    metadata = analysis.get("generation_metadata", {})

    st.caption(
        " · ".join(
            [
                f"Model: {_safe_text(metadata.get('model', 'unknown'))}",
                f"Provider: {_safe_text(metadata.get('provider', 'unknown'))}",
                (
                    "Generation: "
                    f"{float(metadata.get('latency_seconds', 0.0)):.2f}s"
                ),
                (
                    "End-to-end: "
                    f"{format_seconds(st.session_state.get('last_rag_duration'))}"
                ),
            ]
        )
    )


def render_ai_analysis(result: dict) -> None:
    """Render optional, grounded RAG analysis for the current RCA result."""

    section("AI investigation · retrieval-augmented analysis")
    _render_ai_header()

    rag_is_stale = bool(st.session_state.get("rca_stale"))

    if rag_is_stale:
        st.caption(
            "Refresh RCA after the telemetry wait period before generating "
            "a new AI analysis."
        )

    if st.button(
        "✦ GENERATE AI INCIDENT ANALYSIS",
        type="primary",
        width="stretch",
        disabled=rag_is_stale,
        key="generate_ai_incident_analysis",
    ):
        started = time.perf_counter()

        try:
            with st.spinner(
                "Retrieving validated incidents · ranking evidence · "
                "generating grounded incident analysis"
            ):
                from rootlens.rag.explainability import generate_rca_explanation

                st.session_state.rag_analysis = generate_rca_explanation(result)
                st.session_state.rag_source_timestamp = result["timestamp"]

            st.session_state.rag_error = None

        except Exception as exc:
            LOGGER.exception("RootLens AI analysis unavailable")
            st.session_state.rag_error = str(exc)

        finally:
            st.session_state.last_rag_duration = time.perf_counter() - started

    analysis = st.session_state.get("rag_analysis")
    source_timestamp = st.session_state.get("rag_source_timestamp")

    if source_timestamp and source_timestamp != result["timestamp"]:
        st.warning(
            "The displayed AI analysis belongs to an earlier RCA refresh. "
            "Generate it again for the current telemetry window."
        )

    if st.session_state.get("rag_error"):
        st.warning(
            "AI analysis unavailable: "
            f"{st.session_state.rag_error}"
        )
        return

    if not analysis:
        st.caption(
            "No AI incident report has been generated for this RCA result yet."
        )
        return

    _render_report(analysis)

    retrieved = analysis.get("retrieved_evidence", {})
    numeric_matches = retrieved.get("numeric_matches", [])
    semantic_matches = retrieved.get("semantic_matches", [])

    with st.expander(
        "RETRIEVED EVIDENCE · VALIDATED INCIDENTS & KNOWLEDGE",
        expanded=False,
    ):
        _render_numeric_matches(numeric_matches)
        st.divider()
        _render_semantic_matches(semantic_matches)
        st.divider()
        _render_generation_metadata(analysis)
