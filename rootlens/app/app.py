#!/usr/bin/env python3
"""RootLensAI technical operations console."""

from __future__ import annotations

import logging
import os
import sys
import time
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

from rootlens.app.console_styles import apply_console_styles
from rootlens.app.rag_ui import render_ai_analysis
from rootlens.app.runtime_ui import (
    cooldown_active,
    cooldown_remaining,
    init_ui_state,
    maybe_tick_countdown,
    render_cooldown_banner,
    render_demo_controls,
    render_runtime_cards,
    render_stale_rca_banner,
    section,
)

from rootlens.app.ui_components import (
    APP_CSS,
    header_html,
    probabilities_html,
    service_status_html,
    status_html,
    telemetry_style,
    topology_svg,
)
from rootlens.app.calibration_ui import (
    render_calibration_panel,
)
from rootlens.inference.live_telemetry import run_live_rca
from rootlens.inference.rca_inference import RCAInference

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("rootlens.mvp")

TOPOLOGY_PATH = REPO_ROOT / "rootlens/config/service_graph_v1.yaml"

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RootLensAI",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Keep your existing shared component CSS, then layer console overrides on top.
st.markdown(APP_CSS, unsafe_allow_html=True)
apply_console_styles()


@st.cache_resource
def load_engine() -> RCAInference:
    """Load the RCA model once per Streamlit process."""
    return RCAInference(device="cpu")


def render_model_details(result: dict) -> None:
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
    predicted = str(result["predicted_root_cause"])
    services = list(result["service_metrics"])

    st.markdown(status_html(result), unsafe_allow_html=True)

    overview, topology = st.columns([0.92, 1.35], gap="large")

    with overview:
        section("Service status")
        st.markdown(
            service_status_html(services, predicted),
            unsafe_allow_html=True,
        )

    with topology:
        section("Service topology · model-indicated root cause")
        st.markdown(
            topology_svg(TOPOLOGY_PATH, predicted),
            unsafe_allow_html=True,
        )

    section("Model output · class probabilities")
    st.markdown(
        probabilities_html(result["probabilities"], predicted),
        unsafe_allow_html=True,
    )

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
        telemetry_style(table, predicted),
        hide_index=True,
        width="stretch",
        height=455,
    )

    render_model_details(result)

    inferred_at = datetime.fromisoformat(
        result["timestamp"].replace("Z", "+00:00")
    )

    source = "DEMO INPUT" if result["demo_mode"] else "PROMETHEUS CONNECTED"

    st.markdown(
        (
            '<div class="rl-footer">'
            'MODEL READY'
            '&nbsp;&nbsp;·&nbsp;&nbsp;'
            'GRAPH [12×7]'
            '&nbsp;&nbsp;·&nbsp;&nbsp;'
            f'{source}'
            '&nbsp;&nbsp;·&nbsp;&nbsp;'
            f'LAST INFERENCE {inferred_at.strftime("%H:%M:%S UTC")}'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


# ===========================================================================
# APPLICATION
# ===========================================================================

default_demo = (
    os.getenv("ROOTLENS_DEMO_MODE", "false").lower()
    in {"1", "true", "yes", "on"}
)

init_ui_state()
maybe_tick_countdown()

mode_column, refresh_column, spacer = st.columns(
    [1.25, 1.8, 3.95],
    gap="medium",
)

with mode_column.container(key="demo_mode_control"):
    demo_mode = st.toggle(
        "DEMO MODE",
        value=default_demo,
        key="rootlens_demo_mode",
    )

remaining = cooldown_remaining()
refresh_label = f"RCA LOCKED · {remaining}s" if remaining else "REFRESH RCA"

with refresh_column.container(key="refresh_rca_action"):
    refresh = st.button(
        refresh_label,
        type="secondary",
        width="stretch",
        disabled=bool(remaining),
    )

st.markdown(
    header_html(demo_mode),
    unsafe_allow_html=True,
)

if demo_mode:
    st.markdown(
        '<div class="rl-mode-note">DEMO TELEMETRY INPUT</div>',
        unsafe_allow_html=True,
    )

render_runtime_cards()
render_cooldown_banner()
render_demo_controls()

section("RCA result · telemetry-derived inference")

stored_result = st.session_state.get("rca_result")

mode_changed = (
    stored_result is not None
    and bool(stored_result.get("demo_mode")) != demo_mode
)

can_infer = not cooldown_active()

should_run_initial = stored_result is None and can_infer
should_run_refresh = bool(refresh) and can_infer
should_run_mode_change = mode_changed and can_infer

if should_run_initial or should_run_refresh or should_run_mode_change:
    started = time.perf_counter()

    try:
        with st.spinner(
            "Collecting telemetry · constructing graph · running GraphSAGE inference"
        ):
            result = run_live_rca(
                demo_mode=demo_mode,
                inference=load_engine(),
            )

        st.session_state.rca_result = result
        st.session_state.rca_error = None
        st.session_state.last_rca_duration = time.perf_counter() - started
        st.session_state.rca_stale = False

        # A fresh RCA result invalidates any explanation generated from an older
        # telemetry snapshot.
        st.session_state.rag_analysis = None
        st.session_state.rag_source_timestamp = None
        st.session_state.rag_error = None
        st.session_state.last_rag_duration = None

    except Exception as exc:
        st.session_state.last_rca_duration = time.perf_counter() - started
        LOGGER.exception("RootLens RCA refresh failed")
        st.session_state.rca_error = str(exc)

is_stale = bool(st.session_state.get("rca_stale"))

if is_stale:
    render_stale_rca_banner()

    if cooldown_active():
        st.info(
            "The previous RCA result is hidden because the testbed changed. "
            "Wait for the telemetry countdown to finish, then click Refresh RCA."
        )
    else:
        st.info(
            "The telemetry wait period has elapsed. Click Refresh RCA to run "
            "a fresh inference for the current testbed state."
        )

elif st.session_state.get("rca_error"):
    st.error(f"RCA unavailable: {st.session_state.rca_error}")

elif st.session_state.get("rca_result"):
    current_result = (st.session_state.rca_result)
    render_result(st.session_state.rca_result)
    render_calibration_panel(current_result)
    render_ai_analysis(st.session_state.rca_result)

elif cooldown_active():
    st.info(
        "RCA has not been run yet. Wait for the telemetry countdown to finish, "
        "then click Refresh RCA."
    )
