"""RootLensAI Streamlit console styling."""

from __future__ import annotations

import streamlit as st

CONSOLE_CSS = r"""
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
"""


def apply_console_styles() -> None:
    """Apply RootLensAI application styling."""
    st.markdown(CONSOLE_CSS, unsafe_allow_html=True)
