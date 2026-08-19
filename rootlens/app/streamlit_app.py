#!/usr/bin/env python3
"""RootLensAI technical operations console."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Repository setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from rootlens.app.ui_components import (  # noqa: E402
    APP_CSS,
    header_html,
    probabilities_html,
    service_status_html,
    status_html,
    telemetry_style,
    topology_svg,
)

from rootlens.demo.fault_controller import (  # noqa: E402
    FaultControllerError,
    get_fault_state,
    inject_fault,
    restore_fault,
    supported_faults,
)

from rootlens.inference.live_telemetry import run_live_rca  # noqa: E402
from rootlens.inference.rca_inference import RCAInference  # noqa: E402


# ---------------------------------------------------------------------------
# Logging / constants
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("rootlens.mvp")

TOPOLOGY_PATH = REPO_ROOT / "rootlens/config/service_graph_v1.yaml"


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RootLensAI",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Existing shared application CSS.
st.markdown(APP_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MVP visual overrides
# ---------------------------------------------------------------------------
#
# These overrides intentionally live in this file so we do not have to modify
# ui_components.py just to improve the console styling.
#
# The important part is that ALL Streamlit buttons are forced to transparent
# backgrounds with borders. The injection-state component is also explicitly
# reset because the original CSS was causing the white blocks visible in the
# screenshot.
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>

/* ==========================================================================
   ROOTLENSAI — MVP CONSOLE OVERRIDES
   ========================================================================== */


/* --------------------------------------------------------------------------
   Global page
   -------------------------------------------------------------------------- */

html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background:
        radial-gradient(
            circle at 18% -10%,
            rgba(94, 113, 255, 0.075),
            transparent 27rem
        ),
        #050505 !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent !important;
}

.block-container {
    max-width: 1540px !important;
    padding-top: 2.1rem !important;
    padding-bottom: 4rem !important;
    padding-left: 2.6rem !important;
    padding-right: 2.6rem !important;
}


/* --------------------------------------------------------------------------
   General typography
   -------------------------------------------------------------------------- */

html,
body,
p,
span,
label,
div {
    -webkit-font-smoothing: antialiased;
}

p {
    color: #c9ccd5;
}


/* --------------------------------------------------------------------------
   Streamlit default chrome cleanup
   -------------------------------------------------------------------------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}


/* --------------------------------------------------------------------------
   Top row
   Demo Mode + Refresh RCA
   -------------------------------------------------------------------------- */

[data-testid="stHorizontalBlock"]:first-of-type {
    align-items: center !important;
}


/* --------------------------------------------------------------------------
   ALL BUTTONS
   -------------------------------------------------------------------------- */

/*
    This is the main fix.

    It applies to:
    - Refresh RCA
    - Inject Fault
    - Restore System
*/

div[data-testid="stButton"] > button,
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stButton"] > button[kind="secondary"] {

    width: 100% !important;
    min-height: 46px !important;

    background: rgba(255, 255, 255, 0.015) !important;
    background-color: transparent !important;

    border: 1px solid #333844 !important;
    border-radius: 7px !important;

    color: #e8eaf0 !important;

    font-family:
        "SFMono-Regular",
        Consolas,
        "Liberation Mono",
        Menlo,
        monospace !important;

    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.065em !important;
    text-transform: uppercase !important;

    padding: 0.72rem 1.15rem !important;

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.025)
        !important;

    transition:
        border-color 150ms ease,
        background-color 150ms ease,
        transform 150ms ease,
        box-shadow 150ms ease !important;
}


/* Remove Streamlit primary-button white styling */
div[data-testid="stButton"] > button[kind="primary"] {
    color: #eef0ff !important;
    border-color: #505a91 !important;
}


/* Button text/icons */
div[data-testid="stButton"] > button p,
div[data-testid="stButton"] > button span {
    color: inherit !important;
}


/* Hover state */
div[data-testid="stButton"] > button:hover {

    background: rgba(116, 124, 255, 0.055) !important;

    border-color: #6873c6 !important;

    color: #ffffff !important;

    box-shadow:
        0 0 0 1px rgba(104, 115, 198, 0.08),
        0 5px 22px rgba(0, 0, 0, 0.22)
        !important;

    transform: translateY(-1px);
}


/* Active click */
div[data-testid="stButton"] > button:active {
    transform: translateY(0);
    background: rgba(116, 124, 255, 0.085) !important;
}


/* Focus */
div[data-testid="stButton"] > button:focus:not(:active) {
    border-color: #707cda !important;
    box-shadow:
        0 0 0 2px rgba(112, 124, 218, 0.10)
        !important;
}


/* Disabled */
div[data-testid="stButton"] > button:disabled {
    opacity: 0.42;
}


/* --------------------------------------------------------------------------
   Toggle
   -------------------------------------------------------------------------- */

[data-testid="stToggle"] {
    padding-top: 0.2rem;
}

[data-testid="stToggle"] label {
    color: #dfe2e9 !important;
    font-size: 0.86rem !important;
}


/* --------------------------------------------------------------------------
   Inputs / selectboxes
   -------------------------------------------------------------------------- */

[data-testid="stSelectbox"] label {
    color: #aeb3bf !important;

    font-size: 0.78rem !important;
    font-weight: 500 !important;

    margin-bottom: 0.38rem !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div {

    background: #111218 !important;

    border: 1px solid #292c35 !important;
    border-radius: 7px !important;

    min-height: 46px !important;

    box-shadow: none !important;

    transition:
        border-color 140ms ease,
        background-color 140ms ease !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
    border-color: #424754 !important;
    background: #14151c !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] * {
    color: #e6e8ee !important;
}


/* Dropdown menu */
div[data-baseweb="popover"] {
    background: #111218 !important;
}

div[data-baseweb="menu"] {
    background: #111218 !important;
    border: 1px solid #30333c !important;
}

div[data-baseweb="menu"] li:hover {
    background: #1c1e27 !important;
}


/* --------------------------------------------------------------------------
   Section labels
   -------------------------------------------------------------------------- */

.rl-section-label {
    color: #858c9c !important;

    font-family:
        "SFMono-Regular",
        Consolas,
        "Liberation Mono",
        Menlo,
        monospace;

    font-size: 0.72rem !important;
    font-weight: 700 !important;

    letter-spacing: 0.13em !important;
    text-transform: uppercase !important;

    margin-top: 1.7rem !important;
    margin-bottom: 0.9rem !important;
}


/* --------------------------------------------------------------------------
   Fault-injection explanatory text
   -------------------------------------------------------------------------- */

.rl-separation-note {

    color: #aeb2bd !important;

    font-size: 0.88rem !important;
    line-height: 1.6 !important;

    padding: 0 0 0.95rem 0 !important;

    border: 0 !important;
    background: transparent !important;
}

.rl-separation-note b {
    color: #eef0f5 !important;
    font-weight: 700 !important;
}


/* --------------------------------------------------------------------------
   Fault injection state

   IMPORTANT:
   Explicit background override removes the white rectangles from the
   screenshot even if APP_CSS defines a light background.
   -------------------------------------------------------------------------- */

.rl-injection-state {

    display: flex !important;
    align-items: center !important;

    gap: 0.85rem !important;

    width: fit-content !important;
    min-width: 230px !important;

    margin-top: 0.7rem !important;
    margin-bottom: 1rem !important;

    padding: 0.65rem 0.9rem !important;

    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important;

    border: 1px solid #292d36 !important;
    border-radius: 6px !important;

    box-shadow: none !important;
}

.rl-injection-state span {

    color: #777e8d !important;

    font-family:
        "SFMono-Regular",
        Consolas,
        "Liberation Mono",
        Menlo,
        monospace;

    font-size: 0.68rem !important;
    font-weight: 700 !important;

    letter-spacing: 0.09em !important;
}

.rl-injection-state b {

    color: #d7dbe4 !important;

    background: transparent !important;

    font-family:
        "SFMono-Regular",
        Consolas,
        "Liberation Mono",
        Menlo,
        monospace;

    font-size: 0.74rem !important;

    letter-spacing: 0.055em !important;
}


/* Active injection */
.rl-injection-state.active {

    background: rgba(239, 149, 66, 0.035) !important;

    border-color: rgba(239, 149, 66, 0.52) !important;

    box-shadow:
        inset 3px 0 0 rgba(239, 149, 66, 0.72)
        !important;
}

.rl-injection-state.active b {
    color: #f0aa67 !important;
}


/* --------------------------------------------------------------------------
   Alerts
   -------------------------------------------------------------------------- */

[data-testid="stAlert"] {

    background: #0d0f13 !important;

    border: 1px solid #2b303a !important;
    border-radius: 7px !important;

    color: #dce0e8 !important;
}


/* --------------------------------------------------------------------------
   Spinner
   -------------------------------------------------------------------------- */

[data-testid="stSpinner"] {
    color: #9da5ff !important;
}


/* --------------------------------------------------------------------------
   DataFrame
   -------------------------------------------------------------------------- */

[data-testid="stDataFrame"] {

    border: 1px solid #23262e !important;
    border-radius: 7px !important;

    overflow: hidden !important;

    background: #090a0d !important;
}


/* --------------------------------------------------------------------------
   Expander
   -------------------------------------------------------------------------- */

[data-testid="stExpander"] {

    background: #090a0d !important;

    border: 1px solid #24272f !important;
    border-radius: 7px !important;

    overflow: hidden;
}

[data-testid="stExpander"] details summary {

    color: #aeb4c0 !important;

    font-family:
        "SFMono-Regular",
        Consolas,
        "Liberation Mono",
        Menlo,
        monospace;

    font-size: 0.73rem !important;
    font-weight: 700 !important;

    letter-spacing: 0.08em !important;
}


/* --------------------------------------------------------------------------
   Code / inline model information
   -------------------------------------------------------------------------- */

code {

    background: rgba(112, 121, 255, 0.055) !important;

    color: #cdd1ff !important;

    border: 1px solid rgba(112, 121, 255, 0.13);

    border-radius: 4px;

    padding: 0.1rem 0.32rem;
}


/* --------------------------------------------------------------------------
   Mode badge
   -------------------------------------------------------------------------- */

.rl-mode-note {

    display: inline-flex !important;

    align-items: center !important;

    padding: 0.38rem 0.66rem !important;

    margin-top: 0.7rem !important;
    margin-bottom: 0.3rem !important;

    background: rgba(114, 124, 255, 0.045) !important;

    border: 1px solid rgba(114, 124, 255, 0.28) !important;
    border-radius: 5px !important;

    color: #9ea6ff !important;

    font-family:
        "SFMono-Regular",
        Consolas,
        "Liberation Mono",
        Menlo,
        monospace;

    font-size: 0.67rem !important;
    font-weight: 700 !important;

    letter-spacing: 0.1em !important;
}


/* --------------------------------------------------------------------------
   Footer
   -------------------------------------------------------------------------- */

.rl-footer {

    margin-top: 1.6rem !important;

    padding-top: 1rem !important;

    border-top: 1px solid #1e2127 !important;

    color: #686f7d !important;

    font-family:
        "SFMono-Regular",
        Consolas,
        "Liberation Mono",
        Menlo,
        monospace;

    font-size: 0.66rem !important;

    letter-spacing: 0.055em !important;
}


/* --------------------------------------------------------------------------
   Horizontal divider consistency
   -------------------------------------------------------------------------- */

hr {
    border-color: #20232a !important;
}


/* --------------------------------------------------------------------------
   Responsive behavior
   -------------------------------------------------------------------------- */

@media (max-width: 1000px) {

    .block-container {
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    .rl-injection-state {
        width: 100% !important;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached inference engine
# ---------------------------------------------------------------------------

@st.cache_resource
def load_engine() -> RCAInference:
    """Load the RCA inference engine once per Streamlit process."""
    return RCAInference(device="cpu")


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    """Render a RootLens console section heading."""
    st.markdown(
        f'<div class="rl-section-label">{title}</div>',
        unsafe_allow_html=True,
    )


def render_model_details(result: dict) -> None:
    """Render model metadata and runtime information."""

    with st.expander("MODEL DETAILS"):

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown(
                """
**Model**

`GraphSAGE RCA v2 node-preserving`

**Dataset**

`RootLens Dataset v2`

**Parameters**

`61,445`

**Input**

`12 services × 7 telemetry features`
                """
            )

        with right:
            st.markdown(
                f"""
**Topology**

`44 message-passing edges`

**MLflow Run**

`c5b8c95fd91d4ce2ab27df44e846ddec`

**Graph Shape**

`{result['graph_shape']}`

**Runtime**

`{result['device']} · window {result['promql_window']}`
                """
            )


def render_result(result: dict) -> None:
    """Render the complete RCA inference result."""

    predicted = str(result["predicted_root_cause"])
    services = list(result["service_metrics"])

    # -----------------------------------------------------------------------
    # Main RCA status
    # -----------------------------------------------------------------------

    st.markdown(
        status_html(result),
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Service overview + topology
    # -----------------------------------------------------------------------

    overview, topology = st.columns(
        [0.92, 1.35],
        gap="large",
    )

    with overview:

        section("Service status")

        st.markdown(
            service_status_html(
                services,
                predicted,
            ),
            unsafe_allow_html=True,
        )

    with topology:

        section("Service topology · model-indicated root cause")

        st.markdown(
            topology_svg(
                TOPOLOGY_PATH,
                predicted,
            ),
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # Probabilities
    # -----------------------------------------------------------------------

    section("Model output · class probabilities")

    st.markdown(
        probabilities_html(
            result["probabilities"],
            predicted,
        ),
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Telemetry
    # -----------------------------------------------------------------------

    section("Live service telemetry")

    table = pd.DataFrame.from_dict(
        result["service_metrics"],
        orient="index",
    )

    table.index.name = "service"

    table = table.reset_index()[
        [
            "service",
            "cpu",
            "memory",
            "request_rate",
            "has_requests",
            "latency_ms",
            "error_rps",
            "error_rate",
        ]
    ]

    st.dataframe(
        telemetry_style(
            table,
            predicted,
        ),
        hide_index=True,
        width="stretch",
        height=455,
    )

    # -----------------------------------------------------------------------
    # Model metadata
    # -----------------------------------------------------------------------

    render_model_details(result)

    # -----------------------------------------------------------------------
    # Runtime footer
    # -----------------------------------------------------------------------

    inferred_at = datetime.fromisoformat(
        result["timestamp"].replace(
            "Z",
            "+00:00",
        )
    )

    source = (
        "DEMO INPUT"
        if result["demo_mode"]
        else "PROMETHEUS CONNECTED"
    )

    st.markdown(
        '<div class="rl-footer">'
        'MODEL READY'
        '&nbsp;&nbsp;·&nbsp;&nbsp;'
        'GRAPH [12×7]'
        '&nbsp;&nbsp;·&nbsp;&nbsp;'
        f'{source}'
        '&nbsp;&nbsp;·&nbsp;&nbsp;'
        f'LAST INFERENCE {inferred_at.strftime("%H:%M:%S UTC")}'
        '</div>',
        unsafe_allow_html=True,
    )


def render_ai_analysis(result: dict) -> None:
    """Render optional, on-demand RAG explanation without affecting RCA."""

    section("AI incident analysis")
    st.caption(
        "Generated by an LLM from local RootLens retrieval evidence. "
        "The RCA prediction is produced independently by GraphSAGE."
    )
    if st.button("GENERATE AI ANALYSIS", width="stretch"):
        try:
            with st.spinner("Retrieving historical evidence · generating grounded explanation"):
                from rootlens.rag.explainability import generate_rca_explanation

                st.session_state.rag_analysis = generate_rca_explanation(result)
                st.session_state.rag_source_timestamp = result["timestamp"]
            st.session_state.rag_error = None
        except Exception as exc:
            LOGGER.error("RootLens AI analysis unavailable: %s", exc)
            st.session_state.rag_error = str(exc)

    analysis = st.session_state.get("rag_analysis")
    source_timestamp = st.session_state.get("rag_source_timestamp")
    if source_timestamp and source_timestamp != result["timestamp"]:
        st.warning("The displayed AI analysis belongs to an earlier RCA refresh. Generate it again for the current window.")
    if st.session_state.get("rag_error"):
        st.warning(f"AI analysis unavailable: {st.session_state.rag_error}")
        return
    if not analysis:
        st.caption("AI generation is on demand and has not been requested for this RCA result.")
        return

    explanation = analysis["explanation"]
    st.markdown("### Summary")
    st.write(explanation["summary"])
    st.markdown("### Supporting Evidence")
    for claim in explanation["evidence"]:
        references = ", ".join(claim.get("evidence_ids", []))
        st.markdown(f"- {claim.get('claim', '')}  `{references}`")
    st.markdown("### Uncertainty")
    st.write(explanation["uncertainty"])
    st.markdown("### Investigate Next")
    for item in explanation["investigate_next"]:
        st.markdown(f"- {item}")

    with st.expander("RETRIEVED EVIDENCE"):
        st.markdown("#### Similar validated incidents")
        for match in analysis["retrieved_evidence"]["numeric_matches"]:
            st.markdown(
                f"**{match['run_id']}** · similarity `{match['similarity']:.3f}` · "
                f"root cause `{match['root_cause']}` · fault `{match['fault_type']}` · split `{match['split']}`"
            )
        st.markdown("#### Knowledge / context")
        for document in analysis["retrieved_evidence"]["semantic_matches"]:
            st.markdown(
                f"**{document['title']}** · similarity `{document['similarity']:.3f}`  \n"
                f"{document['text']}"
            )
        metadata = analysis["generation_metadata"]
        st.caption(
            f"Model: {metadata['model']} · Provider: {metadata['provider']} · "
            f"Generation latency: {metadata['latency_seconds']:.2f}s"
        )


def render_demo_controls() -> None:
    """Render allow-listed testbed fault-injection controls."""

    section("Demo controls · fault injection")

    st.markdown(
        """
        <div class="rl-separation-note">
            <b>FAULT INJECTION</b>
            changes the testbed only. Injected fault metadata is not passed
            to the RCA model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Fault selection
    # -----------------------------------------------------------------------

    choices = supported_faults()

    control_left, control_right = st.columns(
        2,
        gap="medium",
    )

    service = control_left.selectbox(
        "Target service",
        list(choices),
        format_func=lambda value: (
            value
            .replace("_", " ")
            .title()
        ),
    )

    fault_type = control_right.selectbox(
        "Fault type",
        choices[service],
        format_func=str.title,
    )

    # -----------------------------------------------------------------------
    # Current injection state
    # -----------------------------------------------------------------------

    state = get_fault_state()

    if state["active"]:

        state_text = (
            f'{state["service"].replace("_", " ").upper()}'
            f' · {state["fault_type"].upper()}'
            f' · {state["display_value"]}'
        )

        state_class = "active"

    else:

        state_text = "NONE"
        state_class = ""

    st.markdown(
        f"""
        <div class="rl-injection-state {state_class}">
            <span>INJECTION STATE</span>
            <b>{state_text}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Previous operation notification
    # -----------------------------------------------------------------------

    notice = st.session_state.pop(
        "fault_notice",
        None,
    )

    if notice:
        st.success(notice)

    # -----------------------------------------------------------------------
    # Action buttons
    #
    # Wider than before:
    # previously [1, 1, 2]
    # now [1.35, 1.35, 2.3]
    # -----------------------------------------------------------------------

    inject_column, restore_column, _ = st.columns(
        [1.35, 1.35, 2.3],
        gap="medium",
    )

    # -----------------------------------------------------------------------
    # Inject fault
    # -----------------------------------------------------------------------

    if inject_column.button(
        "INJECT FAULT",
        width="stretch",
    ):

        try:

            with st.spinner(
                "Applying allow-listed fault · "
                "recreating target service · "
                "verifying runtime control"
            ):

                inject_fault(
                    service,
                    fault_type,
                )

            st.session_state.fault_notice = (
                "Fault injected and verified. "
                "Allow telemetry to update, then run Refresh RCA "
                "(recommended wait: 30–45 seconds)."
            )

            st.rerun()

        except FaultControllerError as exc:

            LOGGER.exception(
                "Fault injection failed"
            )

            st.error(
                f"Fault injection failed: {exc}"
            )

    # -----------------------------------------------------------------------
    # Restore system
    # -----------------------------------------------------------------------

    if restore_column.button(
        "RESTORE SYSTEM",
        width="stretch",
    ):

        try:

            with st.spinner(
                "Restoring all demo controls · "
                "verifying healthy values"
            ):

                restore_fault()

            st.session_state.fault_notice = (
                "All supported fault controls restored "
                "and verified at healthy values."
            )

            st.rerun()

        except FaultControllerError as exc:

            LOGGER.exception(
                "System restoration failed"
            )

            st.error(
                f"System restoration failed: {exc}"
            )


# ===========================================================================
# APPLICATION
# ===========================================================================


# ---------------------------------------------------------------------------
# Determine initial telemetry mode
# ---------------------------------------------------------------------------

default_demo = (
    os.getenv(
        "ROOTLENS_DEMO_MODE",
        "false",
    ).lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)


# ---------------------------------------------------------------------------
# Top command bar
# ---------------------------------------------------------------------------

mode_column, refresh_column, spacer = st.columns(
    [1.0, 1.45, 4.55],
    gap="medium",
)

demo_mode = mode_column.toggle(
    "Demo mode",
    value=default_demo,
)

refresh = refresh_column.button(
    "REFRESH RCA",
    type="primary",
    width="stretch",
)


# ---------------------------------------------------------------------------
# Application header
# ---------------------------------------------------------------------------

st.markdown(
    header_html(demo_mode),
    unsafe_allow_html=True,
)


if demo_mode:

    st.markdown(
        '<div class="rl-mode-note">'
        'DEMO TELEMETRY INPUT'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Fault controls
# ---------------------------------------------------------------------------

render_demo_controls()


# ---------------------------------------------------------------------------
# RCA inference
# ---------------------------------------------------------------------------

section(
    "RCA result · telemetry-derived inference"
)

stored_result = st.session_state.get(
    "rca_result"
)

mode_changed = (
    stored_result is not None
    and bool(
        stored_result.get("demo_mode")
    )
    != demo_mode
)


# Run inference:
#   1. on initial page load
#   2. when Refresh RCA is pressed
#   3. when Demo/Live mode changes

if (
    stored_result is None
    or refresh
    or mode_changed
):

    try:

        with st.spinner(
            "Collecting telemetry · "
            "constructing graph · "
            "running inference"
        ):

            st.session_state.rca_result = (
                run_live_rca(
                    demo_mode=demo_mode,
                    inference=load_engine(),
                )
            )

        st.session_state.rca_error = None

    except Exception as exc:

        LOGGER.exception(
            "RootLens RCA refresh failed"
        )

        st.session_state.rca_error = str(exc)


# ---------------------------------------------------------------------------
# Render inference result / error
# ---------------------------------------------------------------------------

if st.session_state.get(
    "rca_error"
):

    st.error(
        "RCA unavailable: "
        f'{st.session_state.rca_error}'
    )

elif st.session_state.get(
    "rca_result"
):

    render_result(
        st.session_state.rca_result
    )

    render_ai_analysis(
        st.session_state.rca_result
    )
