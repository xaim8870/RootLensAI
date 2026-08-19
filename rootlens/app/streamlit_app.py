#!/usr/bin/env python3
"""RootLensAI technical operations console."""

from __future__ import annotations

import html
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # Graceful fallback; install command is shown in the UI.
    st_autorefresh = None


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

# RCA cooldowns deliberately exceed / match the 30-second PromQL observation
# window so users do not accidentally interpret a mixed pre/post-change window.
INJECT_RCA_LOCK_SECONDS = 30
RESTORE_RCA_LOCK_SECONDS = 60
COUNTDOWN_REFRESH_MS = 1000


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
# UX v2 styling: timers, stale state, RAG panel, typography
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>

/* ==========================================================================
   ROOTLENSAI — UX V2
   ========================================================================== */

:root {
    --rl-bg: #050505;
    --rl-panel: #0a0c11;
    --rl-panel-2: #0d1018;
    --rl-panel-ai: #0b0d18;
    --rl-border: #252a35;
    --rl-border-strong: #353d50;
    --rl-text: #f3f5f9;
    --rl-muted: #8f98aa;
    --rl-blue: #8ea1ff;
    --rl-violet: #a992ff;
    --rl-orange: #f0a35b;
    --rl-green: #63d8a2;
    --rl-red: #ff6f6f;
}

/* Better application font hierarchy */
html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    font-family:
        Inter,
        "Segoe UI Variable",
        "Segoe UI",
        -apple-system,
        BlinkMacSystemFont,
        sans-serif !important;
}

h1, h2, h3,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    font-family:
        Inter,
        "Segoe UI Variable",
        "Segoe UI",
        -apple-system,
        BlinkMacSystemFont,
        sans-serif !important;
    letter-spacing: -0.02em;
}

.rl-section-label,
.rl-footer,
.rl-mode-note,
.rl-injection-state,
.rl-timer-value,
.rl-operation-label,
.rl-chip,
.rl-ai-kicker,
.rl-ai-metadata {
    font-family:
        "SFMono-Regular",
        Consolas,
        "Liberation Mono",
        Menlo,
        monospace !important;
}

/* --------------------------------------------------------------------------
   Timing / process cards
   -------------------------------------------------------------------------- */

.rl-runtime-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.7rem;
    margin: 0.65rem 0 1rem 0;
}

.rl-runtime-card {
    min-height: 78px;
    padding: 0.82rem 0.9rem;
    border: 1px solid #222733;
    border-radius: 9px;
    background:
        linear-gradient(180deg, rgba(255,255,255,0.018), transparent),
        #080a0f;
}

.rl-runtime-card.active {
    border-color: rgba(142, 161, 255, 0.42);
    background:
        radial-gradient(circle at 100% 0%, rgba(142,161,255,0.11), transparent 55%),
        #090b12;
}

.rl-operation-label {
    color: #727c8f;
    font-size: 0.61rem;
    font-weight: 800;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}

.rl-runtime-card strong {
    display: block;
    margin-top: 0.35rem;
    color: #f5f7fb;
    font-size: 1.05rem;
    font-weight: 750;
}

.rl-runtime-card small {
    display: block;
    margin-top: 0.2rem;
    color: #7e8798;
    font-size: 0.72rem;
}

.rl-countdown {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;

    margin: 0.65rem 0 1rem 0;
    padding: 0.85rem 1rem;

    border: 1px solid rgba(142, 161, 255, 0.34);
    border-radius: 9px;

    background:
        linear-gradient(90deg, rgba(80, 96, 191, 0.12), rgba(80, 96, 191, 0.03)),
        #080a10;
}

.rl-countdown.ready {
    border-color: rgba(99, 216, 162, 0.36);
    background:
        linear-gradient(90deg, rgba(99, 216, 162, 0.08), transparent),
        #080a0d;
}

.rl-countdown-copy {
    min-width: 0;
}

.rl-countdown-title {
    color: #edf0f6;
    font-size: 0.84rem;
    font-weight: 750;
}

.rl-countdown-subtitle {
    margin-top: 0.18rem;
    color: #7f899b;
    font-size: 0.73rem;
    line-height: 1.45;
}

.rl-timer-value {
    flex: 0 0 auto;
    color: #aeb9ff;
    font-size: 1rem;
    font-weight: 850;
    letter-spacing: 0.06em;
}

.rl-countdown.ready .rl-timer-value {
    color: #76dfb0;
}

/* Stale RCA banner */
.rl-stale-banner {
    margin: 0.35rem 0 1rem 0;
    padding: 0.9rem 1rem;

    border: 1px solid rgba(240, 163, 91, 0.42);
    border-left: 3px solid #f0a35b;
    border-radius: 8px;

    background:
        linear-gradient(90deg, rgba(240,163,91,0.09), transparent 72%),
        #090908;
}

.rl-stale-banner b {
    color: #f3b477;
}

.rl-stale-banner span {
    color: #9b927f;
    font-size: 0.78rem;
}

/* Operation success line */
.rl-operation-result {
    display: inline-flex;
    align-items: center;
    gap: 0.65rem;

    margin: 0.25rem 0 0.9rem 0;
    padding: 0.55rem 0.75rem;

    border: 1px solid #28313a;
    border-radius: 7px;
    background: #090c0f;

    color: #9ba5b5;
    font-size: 0.75rem;
}

.rl-operation-result strong {
    color: #e7eaf0;
}

.rl-operation-result .ok {
    color: #6dd6a4;
}

/* --------------------------------------------------------------------------
   AI / RAG investigation panel
   -------------------------------------------------------------------------- */

.rl-ai-shell {
    position: relative;
    overflow: hidden;

    margin-top: 0.4rem;
    margin-bottom: 0.9rem;
    padding: 1.25rem 1.3rem;

    border: 1px solid rgba(145, 126, 255, 0.35);
    border-radius: 12px;

    background:
        radial-gradient(circle at 92% -20%, rgba(139, 112, 255, 0.18), transparent 26rem),
        radial-gradient(circle at 8% 120%, rgba(74, 116, 255, 0.08), transparent 24rem),
        #090b14;

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.025),
        0 10px 40px rgba(0,0,0,0.18);
}

.rl-ai-shell::before {
    content: "";
    position: absolute;
    inset: 0 auto 0 0;
    width: 3px;
    background: linear-gradient(180deg, #8ea1ff, #a992ff);
}

.rl-ai-kicker {
    color: #a7b4ff;
    font-size: 0.64rem;
    font-weight: 850;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.rl-ai-title {
    margin-top: 0.38rem;
    color: #f5f6fb;
    font-size: 1.18rem;
    font-weight: 780;
    letter-spacing: -0.02em;
}

.rl-ai-copy {
    max-width: 900px;
    margin-top: 0.42rem;
    color: #98a1b4;
    font-size: 0.82rem;
    line-height: 1.55;
}

.rl-ai-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.8rem;
}

.rl-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;

    padding: 0.35rem 0.52rem;

    border: 1px solid rgba(145, 126, 255, 0.22);
    border-radius: 999px;

    background: rgba(145, 126, 255, 0.055);
    color: #b9c0ff;

    font-size: 0.61rem;
    font-weight: 750;
    letter-spacing: 0.055em;
}

/* AI generation button — intentionally more prominent */
div[data-testid="stButton"] > button[kind="primary"] {
    background:
        linear-gradient(
            100deg,
            rgba(102, 119, 255, 0.18),
            rgba(147, 111, 255, 0.15)
        ) !important;
    border-color: rgba(139, 139, 255, 0.58) !important;
    color: #f4f5ff !important;
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
    background:
        linear-gradient(
            100deg,
            rgba(102, 119, 255, 0.27),
            rgba(147, 111, 255, 0.23)
        ) !important;
    border-color: #8f98ff !important;
    box-shadow:
        0 0 0 1px rgba(143,152,255,0.08),
        0 10px 32px rgba(60, 48, 145, 0.22) !important;
}

/* AI report */
.rl-ai-report {
    margin-top: 0.9rem;
    border: 1px solid rgba(121, 130, 173, 0.28);
    border-radius: 11px;
    background: #080a11;
    overflow: hidden;
}

.rl-ai-report-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;

    padding: 0.85rem 1rem;
    border-bottom: 1px solid #202534;
    background:
        linear-gradient(90deg, rgba(117, 126, 255, 0.08), transparent),
        #0a0d16;
}

.rl-ai-report-header strong {
    color: #eef1fa;
    font-size: 0.82rem;
}

.rl-ai-report-header span {
    color: #7f89a0;
    font-size: 0.68rem;
}

.rl-ai-block {
    padding: 1rem 1.05rem;
    border-bottom: 1px solid #1d2230;
}

.rl-ai-block:last-child {
    border-bottom: 0;
}

.rl-ai-block-label {
    margin-bottom: 0.55rem;
    color: #8f9cff;
    font-family:
        "SFMono-Regular",
        Consolas,
        monospace;
    font-size: 0.63rem;
    font-weight: 850;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.rl-ai-block p {
    margin: 0;
    color: #c9cfdb;
    font-size: 0.84rem;
    line-height: 1.68;
}

.rl-ai-list {
    display: grid;
    gap: 0.48rem;
}

.rl-ai-list-item {
    display: flex;
    gap: 0.65rem;
    align-items: flex-start;

    padding: 0.65rem 0.72rem;

    border: 1px solid #202638;
    border-radius: 7px;
    background: #0b0e16;
}

.rl-ai-list-index {
    flex: 0 0 auto;
    min-width: 29px;
    color: #9ba7ff;
    font-family:
        "SFMono-Regular",
        Consolas,
        monospace;
    font-size: 0.63rem;
    font-weight: 850;
}

.rl-ai-list-copy {
    color: #bcc4d3;
    font-size: 0.8rem;
    line-height: 1.55;
}

.rl-ai-list-copy code {
    font-size: 0.68rem;
}

.rl-evidence-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.7rem;
    margin-top: 0.65rem;
}

.rl-evidence-card {
    min-height: 118px;
    padding: 0.8rem 0.85rem;

    border: 1px solid #242a3a;
    border-radius: 8px;
    background:
        linear-gradient(180deg, rgba(255,255,255,0.015), transparent),
        #090c13;
}

.rl-evidence-card strong {
    display: block;
    color: #e5e8f0;
    font-size: 0.74rem;
    overflow-wrap: anywhere;
}

.rl-evidence-card .score {
    margin-top: 0.42rem;
    color: #a9b4ff;
    font-family:
        "SFMono-Regular",
        Consolas,
        monospace;
    font-size: 0.68rem;
}

.rl-evidence-card .meta {
    margin-top: 0.42rem;
    color: #7f899b;
    font-size: 0.69rem;
    line-height: 1.45;
}

.rl-ai-metadata {
    margin-top: 0.65rem;
    color: #687289;
    font-size: 0.62rem;
    letter-spacing: 0.04em;
}

@media (max-width: 1050px) {
    .rl-runtime-grid,
    .rl-evidence-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 680px) {
    .rl-runtime-grid,
    .rl-evidence-grid {
        grid-template-columns: 1fr;
    }

    .rl-countdown {
        align-items: flex-start;
        flex-direction: column;
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



def init_ui_state() -> None:
    """Initialize session state used by timers and process-duration UX."""

    defaults = {
        "rca_unlock_at": 0.0,
        "rca_lock_reason": "",
        "rca_lock_seconds": 0,
        "rca_stale": False,
        "last_operation_type": "",
        "last_operation_duration": None,
        "last_operation_finished_at": None,
        "last_rca_duration": None,
        "last_rag_duration": None,
        "rag_analysis": None,
        "rag_source_timestamp": None,
        "rag_error": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def cooldown_remaining() -> int:
    """Return the number of whole seconds remaining before RCA can refresh."""

    unlock_at = float(st.session_state.get("rca_unlock_at", 0.0) or 0.0)
    remaining = max(0.0, unlock_at - time.time())
    return int(math.ceil(remaining))


def cooldown_active() -> bool:
    """Whether the current telemetry stabilization/recovery lock is active."""

    return cooldown_remaining() > 0


def start_rca_cooldown(seconds: int, reason: str) -> None:
    """Lock RCA refresh for a telemetry stabilization/recovery period."""

    st.session_state.rca_unlock_at = time.time() + seconds
    st.session_state.rca_lock_reason = reason
    st.session_state.rca_lock_seconds = seconds
    st.session_state.rca_stale = True


def format_seconds(value: float | None) -> str:
    """Format a process duration for the UI."""

    if value is None:
        return "—"
    if value < 1:
        return f"{value:.2f}s"
    return f"{value:.1f}s"


def maybe_tick_countdown() -> None:
    """Refresh the page once per second while a cooldown is active."""

    if not cooldown_active():
        return

    if st_autorefresh is not None:
        st_autorefresh(
            interval=COUNTDOWN_REFRESH_MS,
            limit=None,
            key="rootlens_cooldown_tick",
        )
    else:
        st.warning(
            "Live countdown requires `streamlit-autorefresh`. "
            "Install it with: `pip install streamlit-autorefresh`"
        )


def render_runtime_cards() -> None:
    """Render measured execution durations for the main application processes."""

    operation_name = st.session_state.get("last_operation_type") or "No operation yet"
    operation_duration = st.session_state.get("last_operation_duration")
    rca_duration = st.session_state.get("last_rca_duration")
    rag_duration = st.session_state.get("last_rag_duration")
    remaining = cooldown_remaining()

    cooldown_value = f"{remaining}s" if remaining else "READY"
    cooldown_note = (
        st.session_state.get("rca_lock_reason", "")
        if remaining
        else "RCA refresh is available"
    )

    st.markdown(
        f"""
        <div class="rl-runtime-grid">
            <div class="rl-runtime-card">
                <div class="rl-operation-label">Last testbed operation</div>
                <strong>{html.escape(str(operation_name))}</strong>
                <small>Execution {format_seconds(operation_duration)}</small>
            </div>

            <div class="rl-runtime-card {'active' if remaining else ''}">
                <div class="rl-operation-label">Telemetry gate</div>
                <strong>{cooldown_value}</strong>
                <small>{html.escape(str(cooldown_note))}</small>
            </div>

            <div class="rl-runtime-card">
                <div class="rl-operation-label">Last RCA process</div>
                <strong>{format_seconds(rca_duration)}</strong>
                <small>Prometheus → graph → GraphSAGE</small>
            </div>

            <div class="rl-runtime-card">
                <div class="rl-operation-label">Last AI analysis</div>
                <strong>{format_seconds(rag_duration)}</strong>
                <small>Retrieval + grounded generation</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_cooldown_banner() -> None:
    """Explain the telemetry waiting period and show the live seconds counter."""

    remaining = cooldown_remaining()

    if remaining:
        reason = st.session_state.get("rca_lock_reason", "Telemetry stabilizing")
        total = st.session_state.get("rca_lock_seconds", remaining)

        st.markdown(
            f"""
            <div class="rl-countdown">
                <div class="rl-countdown-copy">
                    <div class="rl-countdown-title">RCA refresh temporarily locked</div>
                    <div class="rl-countdown-subtitle">
                        {html.escape(str(reason))}. This protects the next inference
                        from a mixed pre/post-change PromQL window.
                    </div>
                </div>
                <div class="rl-timer-value">{remaining}s / {total}s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif st.session_state.get("last_operation_type"):
        st.markdown(
            """
            <div class="rl-countdown ready">
                <div class="rl-countdown-copy">
                    <div class="rl-countdown-title">Telemetry observation window ready</div>
                    <div class="rl-countdown-subtitle">
                        The required wait period has elapsed. You can refresh RCA now.
                    </div>
                </div>
                <div class="rl-timer-value">READY</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_stale_rca_banner() -> None:
    """Clearly mark an RCA result produced before the latest testbed change."""

    if not st.session_state.get("rca_stale"):
        return

    remaining = cooldown_remaining()
    suffix = (
        f" New inference unlocks in {remaining}s."
        if remaining
        else " The wait period has elapsed; refresh RCA for the current telemetry window."
    )

    st.markdown(
        f"""
        <div class="rl-stale-banner">
            <b>STALE RCA RESULT</b><br>
            <span>
                The testbed changed after this inference.{html.escape(suffix)}
                The result below is kept only for comparison.
            </span>
        </div>
        """,
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
    """Render a visually distinct, on-demand grounded RAG explanation."""

    section("AI investigation · retrieval-augmented analysis")

    st.markdown(
        """
        <div class="rl-ai-shell">
            <div class="rl-ai-kicker">ROOTLENSAI · GROUNDED INCIDENT INTELLIGENCE</div>
            <div class="rl-ai-title">AI-assisted incident investigation</div>
            <div class="rl-ai-copy">
                Combines the current GraphSAGE RCA output with retrieved validated
                incidents and frozen knowledge context. Fault-injection metadata is
                never supplied to the model or the explanation pipeline.
            </div>
            <div class="rl-ai-chip-row">
                <span class="rl-chip">GRAPH RCA INDEPENDENT</span>
                <span class="rl-chip">LOCAL RETRIEVAL EVIDENCE</span>
                <span class="rl-chip">UNCERTAINTY PRESERVED</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rag_is_stale = bool(st.session_state.get("rca_stale"))
    generate_disabled = rag_is_stale

    if generate_disabled:
        st.caption(
            "Refresh RCA after the telemetry wait period before generating a new AI analysis."
        )

    if st.button(
        "✦ GENERATE AI INCIDENT ANALYSIS",
        type="primary",
        width="stretch",
        disabled=generate_disabled,
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
        st.warning(f"AI analysis unavailable: {st.session_state.rag_error}")
        return

    if not analysis:
        st.caption(
            "No AI incident report has been generated for this RCA result yet."
        )
        return

    explanation = analysis["explanation"]

    summary = html.escape(str(explanation.get("summary", "")))
    uncertainty = html.escape(str(explanation.get("uncertainty", "")))

    evidence_rows = []
    for index, claim in enumerate(explanation.get("evidence", []), start=1):
        references = ", ".join(claim.get("evidence_ids", [])) or "—"
        evidence_rows.append(
            f"""
            <div class="rl-ai-list-item">
                <div class="rl-ai-list-index">E{index:02d}</div>
                <div class="rl-ai-list-copy">
                    {html.escape(str(claim.get("claim", "")))}<br>
                    <code>{html.escape(references)}</code>
                </div>
            </div>
            """
        )

    investigate_rows = []
    for index, item in enumerate(explanation.get("investigate_next", []), start=1):
        investigate_rows.append(
            f"""
            <div class="rl-ai-list-item">
                <div class="rl-ai-list-index">{index:02d}</div>
                <div class="rl-ai-list-copy">{html.escape(str(item))}</div>
            </div>
            """
        )

    metadata = analysis.get("generation_metadata", {})

    st.markdown(
        f"""
        <div class="rl-ai-report">
            <div class="rl-ai-report-header">
                <strong>AI INCIDENT REPORT</strong>
                <span>
                    generated in {format_seconds(st.session_state.get("last_rag_duration"))}
                </span>
            </div>

            <div class="rl-ai-block">
                <div class="rl-ai-block-label">Summary</div>
                <p>{summary}</p>
            </div>

            <div class="rl-ai-block">
                <div class="rl-ai-block-label">Supporting evidence</div>
                <div class="rl-ai-list">
                    {''.join(evidence_rows)}
                </div>
            </div>

            <div class="rl-ai-block">
                <div class="rl-ai-block-label">Uncertainty</div>
                <p>{uncertainty}</p>
            </div>

            <div class="rl-ai-block">
                <div class="rl-ai-block-label">Investigate next</div>
                <div class="rl-ai-list">
                    {''.join(investigate_rows)}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    numeric_matches = analysis.get("retrieved_evidence", {}).get(
        "numeric_matches",
        [],
    )

    with st.expander("RETRIEVED EVIDENCE · VALIDATED INCIDENTS & KNOWLEDGE"):
        st.markdown("#### Similar validated incidents")

        cards = []
        for match in numeric_matches:
            cards.append(
                f"""
                <div class="rl-evidence-card">
                    <strong>{html.escape(str(match.get("run_id", "")))}</strong>
                    <div class="score">
                        similarity {float(match.get("similarity", 0.0)):.3f}
                    </div>
                    <div class="meta">
                        root cause: {html.escape(str(match.get("root_cause", "")))}<br>
                        fault: {html.escape(str(match.get("fault_type", "")))}<br>
                        split: {html.escape(str(match.get("split", "")))}
                    </div>
                </div>
                """
            )

        if cards:
            st.markdown(
                f'<div class="rl-evidence-grid">{"".join(cards)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No numeric incident matches were returned.")

        st.markdown("#### Knowledge / context")

        for document in analysis.get("retrieved_evidence", {}).get(
            "semantic_matches",
            [],
        ):
            st.markdown(
                f"""
**{document.get('title', '')}** · similarity `{document.get('similarity', 0.0):.3f}`

{document.get('text', '')}
                """
            )

        st.markdown(
            f"""
            <div class="rl-ai-metadata">
                MODEL {html.escape(str(metadata.get("model", "unknown")))}
                &nbsp;·&nbsp;
                PROVIDER {html.escape(str(metadata.get("provider", "unknown")))}
                &nbsp;·&nbsp;
                GENERATION {float(metadata.get("latency_seconds", 0.0)):.2f}s
                &nbsp;·&nbsp;
                END-TO-END {format_seconds(st.session_state.get("last_rag_duration"))}
            </div>
            """,
            unsafe_allow_html=True,
        )



def render_demo_controls() -> None:
    """Render controlled testbed fault injection with measured operation timing."""

    section("Demo controls · fault injection")

    st.markdown(
        """
        <div class="rl-separation-note">
            <b>FAULT INJECTION</b>
            changes the testbed only. Injected fault metadata is not passed
            to the RCA model. RCA refresh is temporarily locked after a
            testbed change so the next 30-second PromQL window is interpretable.
        </div>
        """,
        unsafe_allow_html=True,
    )

    choices = supported_faults()

    control_left, control_right = st.columns(
        2,
        gap="medium",
    )

    service = control_left.selectbox(
        "Target service",
        list(choices),
        format_func=lambda value: value.replace("_", " ").title(),
    )

    fault_type = control_right.selectbox(
        "Fault type",
        choices[service],
        format_func=str.title,
    )

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
            <b>{html.escape(state_text)}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    last_operation = st.session_state.get("last_operation_type")
    last_duration = st.session_state.get("last_operation_duration")

    if last_operation and last_duration is not None:
        st.markdown(
            f"""
            <div class="rl-operation-result">
                <span class="ok">●</span>
                <span>
                    <strong>{html.escape(str(last_operation))}</strong>
                    completed in <strong>{format_seconds(last_duration)}</strong>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    notice = st.session_state.pop("fault_notice", None)
    if notice:
        st.success(notice)

    active_wait = cooldown_active()

    inject_column, restore_column, _ = st.columns(
        [1.35, 1.35, 2.3],
        gap="medium",
    )

    inject_disabled = bool(state["active"]) or active_wait
    restore_disabled = not bool(state["active"])

    if inject_column.button(
        "INJECT FAULT",
        width="stretch",
        disabled=inject_disabled,
    ):
        started = time.perf_counter()

        try:
            with st.spinner(
                "Applying allow-listed fault · recreating target service · "
                "verifying runtime control"
            ):
                inject_fault(service, fault_type)

            duration = time.perf_counter() - started

            st.session_state.last_operation_type = (
                f"Injected {service.replace('_', ' ').title()} "
                f"{fault_type.title()}"
            )
            st.session_state.last_operation_duration = duration
            st.session_state.last_operation_finished_at = time.time()

            start_rca_cooldown(
                INJECT_RCA_LOCK_SECONDS,
                (
                    f"Fault injected; waiting {INJECT_RCA_LOCK_SECONDS}s "
                    "for the PromQL window to reflect the new condition"
                ),
            )

            st.session_state.fault_notice = (
                f"Fault injected and verified in {duration:.2f}s. "
                f"RCA unlocks after {INJECT_RCA_LOCK_SECONDS}s."
            )

            st.rerun()

        except FaultControllerError as exc:
            duration = time.perf_counter() - started
            st.session_state.last_operation_type = "Fault injection failed"
            st.session_state.last_operation_duration = duration
            LOGGER.exception("Fault injection failed")
            st.error(
                f"Fault injection failed after {duration:.2f}s: {exc}"
            )

    if restore_column.button(
        "RESTORE SYSTEM",
        width="stretch",
        disabled=restore_disabled,
    ):
        started = time.perf_counter()

        try:
            with st.spinner(
                "Restoring all demo controls · verifying healthy values"
            ):
                restore_fault()

            duration = time.perf_counter() - started

            st.session_state.last_operation_type = "System restore"
            st.session_state.last_operation_duration = duration
            st.session_state.last_operation_finished_at = time.time()

            start_rca_cooldown(
                RESTORE_RCA_LOCK_SECONDS,
                (
                    f"System restored; waiting {RESTORE_RCA_LOCK_SECONDS}s "
                    "for fault residue to leave the rolling telemetry window"
                ),
            )

            st.session_state.fault_notice = (
                f"System restored and verified in {duration:.2f}s. "
                f"RCA unlocks after {RESTORE_RCA_LOCK_SECONDS}s."
            )

            st.rerun()

        except FaultControllerError as exc:
            duration = time.perf_counter() - started
            st.session_state.last_operation_type = "System restore failed"
            st.session_state.last_operation_duration = duration
            LOGGER.exception("System restoration failed")
            st.error(
                f"System restoration failed after {duration:.2f}s: {exc}"
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
# Session state + countdown tick
# ---------------------------------------------------------------------------

init_ui_state()
maybe_tick_countdown()


# ---------------------------------------------------------------------------
# Top command bar
# ---------------------------------------------------------------------------

mode_column, refresh_column, spacer = st.columns(
    [1.0, 1.65, 4.35],
    gap="medium",
)

demo_mode = mode_column.toggle(
    "Demo mode",
    value=default_demo,
)

remaining = cooldown_remaining()
refresh_label = (
    f"RCA LOCKED · {remaining}s"
    if remaining
    else "REFRESH RCA"
)

refresh = refresh_column.button(
    refresh_label,
    type="secondary",
    width="stretch",
    disabled=bool(remaining),
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
        '<div class="rl-mode-note">DEMO TELEMETRY INPUT</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Process timing / telemetry readiness
# ---------------------------------------------------------------------------

render_runtime_cards()
render_cooldown_banner()


# ---------------------------------------------------------------------------
# Fault controls
# ---------------------------------------------------------------------------

render_demo_controls()


# ---------------------------------------------------------------------------
# RCA inference
# ---------------------------------------------------------------------------

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

        # A new RCA window invalidates the previously generated explanation.
        st.session_state.rag_analysis = None
        st.session_state.rag_source_timestamp = None
        st.session_state.rag_error = None
        st.session_state.last_rag_duration = None

    except Exception as exc:
        st.session_state.last_rca_duration = time.perf_counter() - started
        LOGGER.exception("RootLens RCA refresh failed")
        st.session_state.rca_error = str(exc)


# ---------------------------------------------------------------------------
# Render inference result / error
# ---------------------------------------------------------------------------

render_stale_rca_banner()

if st.session_state.get("rca_error"):
    st.error(
        "RCA unavailable: "
        f'{st.session_state.rca_error}'
    )

elif st.session_state.get("rca_result"):
    render_result(st.session_state.rca_result)
    render_ai_analysis(st.session_state.rca_result)

elif cooldown_active():
    st.info(
        "RCA has not been run yet. Wait for the telemetry countdown to finish, "
        "then click Refresh RCA."
    )