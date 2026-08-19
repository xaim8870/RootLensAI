"""Central visual system for the RootLensAI Streamlit console."""

from __future__ import annotations

import streamlit as st


CONSOLE_CSS = r"""
<style>
:root {
    --rl-bg: #050606; --rl-surface: #090b0d; --rl-line: #282d31;
    --rl-line-strong: #41484d; --rl-text: #f1f1ec; --rl-muted: #899097;
    --rl-amber: #d6a84b; --rl-amber-bright: #e1b75a; --rl-amber-dark: #8e6b2d;
    --rl-green: #63bf8b; --rl-red: #ef625b; --rl-violet: #8d86bc;
    --rl-radius: 3px;
}

/* Page and typography */
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
    background: var(--rl-bg) !important; color: var(--rl-text) !important;
    font-family: "Segoe UI Variable", "Segoe UI", Inter, Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
[data-testid="stAppViewContainer"] > .main { background: transparent !important; }
.block-container { max-width: 1500px !important; padding: 1.65rem 2.35rem 3rem !important; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }
p { color: #bcc1c4; }
code, .rl-section-label, .rl-operation-label, .rl-timer-value, .rl-footer,
.rl-badge, .rl-injection-state, .rl-ai-kicker, .rl-chip {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace !important;
}
hr { border-color: #202529 !important; margin: 1rem 0 !important; }

/* Product header */
.rl-header { align-items: end !important; border-bottom: 1px solid var(--rl-line) !important; padding-bottom: 1.05rem !important; }
.rl-brand {
    position: relative; width: fit-content; padding-right: 2.4rem;
    color: var(--rl-text) !important;
    font-family: "Arial Narrow", "Roboto Condensed", "Segoe UI Variable", sans-serif !important;
    font-size: 2.2rem !important; font-stretch: condensed; font-weight: 820 !important;
    letter-spacing: .055em !important; line-height: 1 !important; text-transform: uppercase;
}
.rl-brand::after { content: "AI"; position: absolute; right: 0; bottom: 0; color: var(--rl-amber); font-size: .48em; letter-spacing: .08em; }
.rl-subtitle { margin-top: .45rem !important; color: var(--rl-muted) !important; font-size: .82rem !important; }
.rl-badges { gap: .4rem !important; }
.rl-badge {
    border: 1px solid var(--rl-line) !important; border-bottom: 2px solid #4a4f52 !important;
    border-radius: var(--rl-radius) !important; background: #080a0b !important;
    padding: .38rem .55rem !important; color: #c5c9ca !important; font-size: .61rem !important;
}
.rl-badge b { color: var(--rl-amber) !important; }

/* Native Demo Mode toggle */
.st-key-demo_mode_control {
    position: relative; min-height: 48px; padding: .45rem .7rem !important;
    border: 1px solid var(--rl-line-strong); border-bottom: 3px solid var(--rl-amber-dark);
    border-radius: var(--rl-radius); background: #090b0c;
}
.st-key-demo_mode_control::after {
    content: "OFF"; position: absolute; right: .7rem; top: 50%; transform: translateY(-50%);
    color: var(--rl-muted); font: 800 .62rem/1 "SFMono-Regular", Consolas, monospace;
    letter-spacing: .1em; pointer-events: none;
}
.st-key-demo_mode_control:has(input:checked)::after { content: "ENABLED"; color: var(--rl-amber-bright); }
.st-key-demo_mode_control [data-testid="stToggle"] { padding: 0 !important; }
.st-key-demo_mode_control [data-testid="stToggle"] label { gap: .55rem !important; padding-right: 2.8rem !important; }
.st-key-demo_mode_control [data-testid="stToggle"] p { color: #d5d7d5 !important; font: 800 .64rem/1 "SFMono-Regular", Consolas, monospace !important; letter-spacing: .1em; }
.st-key-demo_mode_control [data-baseweb="checkbox"] > div:first-child {
    width: 31px !important; height: 17px !important; border: 1px solid #4a5052 !important;
    border-radius: 2px !important; background: #111416 !important;
}
.st-key-demo_mode_control [data-baseweb="checkbox"] > div:first-child > div {
    width: 9px !important; height: 9px !important; border-radius: 1px !important; background: #737a7d !important;
}
.st-key-demo_mode_control:has(input:checked) [data-baseweb="checkbox"] > div:first-child { border-color: var(--rl-amber-dark) !important; background: #171309 !important; }
.st-key-demo_mode_control:has(input:checked) [data-baseweb="checkbox"] > div:first-child > div { background: var(--rl-amber) !important; }
.st-key-demo_mode_control [data-testid="stToggle"] label > div:first-of-type {
    width: 30px !important; height: 16px !important; flex: 0 0 30px !important;
    border: 1px solid #4a5052 !important; border-radius: 2px !important;
    background: #111416 !important; padding: 2px !important;
}
.st-key-demo_mode_control [data-testid="stToggle"] label > div:first-of-type > div {
    width: 10px !important; height: 10px !important; border-radius: 1px !important;
    background: #737a7d !important; box-shadow: none !important;
}
.st-key-demo_mode_control:has(input:checked) [data-testid="stToggle"] label > div:first-of-type {
    border-color: var(--rl-amber-dark) !important; background: #171309 !important;
}
.st-key-demo_mode_control:has(input:checked) [data-testid="stToggle"] label > div:first-of-type > div {
    background: var(--rl-amber) !important;
}

/* Button system */
div[data-testid="stButton"] > button,
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stButton"] > button[kind="secondary"] {
    width: 100% !important; min-height: 45px !important; padding: .68rem 1rem !important;
    border: 1px solid #66532d !important; border-bottom: 3px solid var(--rl-amber-dark) !important;
    border-radius: var(--rl-radius) !important; background: #121006 !important; color: #edddb9 !important;
    box-shadow: none !important; font: 800 .7rem/1 "SFMono-Regular", Consolas, monospace !important;
    letter-spacing: .075em !important; text-transform: uppercase !important;
    transition: background-color 120ms ease, border-color 120ms ease, color 120ms ease !important;
}
div[data-testid="stButton"] > button:hover:not(:disabled) {
    transform: none !important; border-color: var(--rl-amber) !important;
    border-bottom-color: var(--rl-amber-bright) !important; background: #1a1609 !important; color: #fff5dc !important;
}
div[data-testid="stButton"] > button:focus:not(:active) { outline: 1px solid var(--rl-amber) !important; outline-offset: 2px !important; box-shadow: none !important; }
div[data-testid="stButton"] > button:disabled {
    opacity: 1 !important; border-color: #2b3032 !important; border-bottom-color: #373d40 !important;
    background: #0b0d0e !important; color: #60676b !important;
}
.st-key-refresh_rca_action button { background: var(--rl-amber) !important; color: #171207 !important; border-color: #e0ba67 !important; border-bottom-color: #8a6729 !important; }
.st-key-refresh_rca_action button:hover:not(:disabled) { background: var(--rl-amber-bright) !important; color: #100d06 !important; }
.st-key-fault_inject_action button { border-left: 3px solid var(--rl-amber) !important; }
.st-key-restore_system_action button { border-color: #355c48 !important; border-bottom-color: #477b5f !important; background: #09130e !important; color: #a8d5ba !important; }
.st-key-restore_system_action button:hover:not(:disabled) { border-color: var(--rl-green) !important; border-bottom-color: var(--rl-green) !important; background: #0c1b13 !important; color: #d5f2e1 !important; }
.st-key-generate_ai_incident_analysis button, .st-key-generate_ai_incident_analysis button[kind="primary"] {
    border-color: #716238 !important; border-top: 2px solid var(--rl-violet) !important;
    border-bottom: 3px solid var(--rl-amber) !important; background: #151207 !important; color: #f0dfb8 !important;
}

/* Inputs */
[data-testid="stSelectbox"] label { color: #a5aaab !important; font-size: .72rem !important; font-weight: 650 !important; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    min-height: 45px !important; border: 1px solid #363c3f !important; border-bottom: 3px solid #50573f !important;
    border-radius: var(--rl-radius) !important; background: #0c0f10 !important; box-shadow: none !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover { border-color: var(--rl-amber-dark) !important; border-bottom-color: var(--rl-amber) !important; }
[data-testid="stSelectbox"] div[data-baseweb="select"] * { color: #e1e2dd !important; }
div[data-baseweb="popover"], div[data-baseweb="menu"] { border-radius: 2px !important; background: #0d1011 !important; }
div[data-baseweb="menu"] { border: 1px solid var(--rl-line-strong) !important; }
div[data-baseweb="menu"] li:hover { background: #1b180e !important; }

/* Section labels and instrumentation */
.rl-section-label {
    margin: 1.35rem 0 .65rem !important; color: #9da29e !important; font-size: .64rem !important;
    font-weight: 850 !important; letter-spacing: .14em !important; text-transform: uppercase !important;
}
.rl-section-label::before { content: ""; display: inline-block; width: 14px; height: 2px; margin: 0 .5rem .18rem 0; background: var(--rl-amber); }
.rl-runtime-grid { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 0; margin: .75rem 0 .9rem; border: 1px solid var(--rl-line); border-left: 3px solid #555b5e; }
.rl-runtime-card {
    min-height: 78px !important; padding: .75rem .85rem !important; border: 0 !important;
    border-right: 1px solid var(--rl-line) !important; border-radius: 0 !important;
    background: #080a0b !important; box-shadow: none !important;
}
.rl-runtime-card:last-child { border-right: 0 !important; }
.rl-runtime-card.active { background: #121006 !important; box-shadow: inset 0 3px 0 var(--rl-amber-dark) !important; }
.rl-operation-label { color: #747c7f !important; font-size: .58rem !important; font-weight: 800 !important; letter-spacing: .11em !important; }
.rl-runtime-card strong { display: block; margin-top: .35rem; color: #e8e9e4; font-size: .98rem; }
.rl-runtime-card small { display: block; margin-top: .2rem; color: #697174; font-size: .69rem; }

/* Countdown and operation state */
.rl-countdown, .rl-stale-banner {
    margin: .6rem 0 .9rem !important; border: 1px solid #5f512f !important;
    border-left: 4px solid var(--rl-amber) !important; border-radius: var(--rl-radius) !important;
    background: #100e08 !important; box-shadow: none !important;
}
.rl-countdown { padding: .75rem .9rem !important; }
.rl-countdown.ready { border-color: #345b47 !important; border-left-color: var(--rl-green) !important; background: #08110c !important; }
.rl-countdown-title { color: #e3e2d9 !important; font-size: .8rem !important; }
.rl-countdown-subtitle { color: #898a80 !important; font-size: .7rem !important; }
.rl-timer-value { color: var(--rl-amber-bright) !important; }.rl-countdown.ready .rl-timer-value { color: var(--rl-green) !important; }
.rl-stale-banner { padding: .8rem .9rem !important; }.rl-stale-banner b { color: var(--rl-amber-bright) !important; }.rl-stale-banner span { color: #9e9278 !important; font-size: .74rem !important; }
.rl-separation-note { margin: 0 0 .7rem !important; padding: .7rem .85rem !important; border: 1px solid var(--rl-line) !important; border-left: 3px solid var(--rl-amber-dark) !important; border-radius: var(--rl-radius) !important; background: #090b0b !important; color: #9ca1a1 !important; font-size: .78rem !important; }
.rl-injection-state { margin: .65rem 0 .8rem !important; padding: .58rem .75rem !important; border: 1px solid var(--rl-line) !important; border-left: 3px solid #50575a !important; border-radius: var(--rl-radius) !important; background: #080a0b !important; box-shadow: none !important; }
.rl-injection-state.active { border-color: #664642 !important; border-left-color: var(--rl-red) !important; background: #130a09 !important; }
.rl-injection-state span { color: #747c7f !important; font-size: .61rem !important; }.rl-injection-state b { color: #dedfd9 !important; font-size: .7rem !important; }.rl-injection-state.active b { color: #ef8b85 !important; }
.rl-operation-result { border: 1px solid #32483c !important; border-left: 3px solid var(--rl-green) !important; border-radius: var(--rl-radius) !important; background: #08100b !important; box-shadow: none !important; }.rl-operation-result .ok { color: var(--rl-green) !important; }

/* RCA status, probabilities, services */
.rl-status { border: 1px solid var(--rl-line) !important; border-left: 4px solid var(--rl-green) !important; border-radius: var(--rl-radius) !important; background: #080a0b !important; }
.rl-status.incident { border-color: #5d3835 !important; border-left-color: var(--rl-red) !important; background: #0d0909 !important; }
.rl-status > div { min-height: 112px !important; padding: 1rem 1.15rem !important; }
.rl-status-state { font-family: "Arial Narrow", "Roboto Condensed", "Segoe UI Variable", sans-serif !important; font-size: 1.55rem !important; letter-spacing: .025em !important; }
.rl-root-value { font-size: 1.7rem !important; }.rl-confidence { font-size: 1.65rem !important; font-variant-numeric: tabular-nums; }
.rl-field-label { color: #777f82 !important; font-family: "SFMono-Regular",Consolas,monospace !important; font-size: .59rem !important; }
.rl-confidence-note { margin-top: .42rem; color: #70787a !important; font-size: .62rem; line-height: 1.4; }
.rl-prob-row { padding: .34rem 0 !important; }.rl-track { height: 5px !important; background: #191d1e !important; }.rl-fill { background: #b5b7b2 !important; }.rl-prob-row.predicted .rl-fill { background: var(--rl-amber) !important; }.rl-prob-row.incident .rl-fill { background: var(--rl-red) !important; }
.rl-service-grid { gap: 0 !important; border-color: var(--rl-line) !important; }
.rl-service { border-color: var(--rl-line) !important; border-radius: 0 !important; background: #080a0b !important; padding: .6rem .7rem !important; }
.rl-service.faulty { border-left: 4px solid var(--rl-red) !important; background: #130908 !important; }.rl-service-name { font-size: .7rem !important; }.rl-service-state { font-size: .57rem !important; }

/* Topology, telemetry, calibration */
.rl-topology-shell { border: 1px solid var(--rl-line) !important; border-left: 3px solid #4e5558 !important; border-radius: var(--rl-radius) !important; background: #070909 !important; padding: .45rem !important; overflow-x: auto; }
[data-testid="stDataFrame"] { border: 1px solid var(--rl-line) !important; border-left: 3px solid #4e5558 !important; border-radius: var(--rl-radius) !important; background: #080a0b !important; overflow: hidden !important; }
[data-testid="stDataFrame"] canvas { font-variant-numeric: tabular-nums; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--rl-line) !important; border-left: 3px solid #52595c !important; border-radius: var(--rl-radius) !important; background: #090b0c !important; box-shadow: none !important; }
[data-testid="stMetric"] { padding: .7rem .8rem !important; border: 1px solid #24292c !important; border-left: 3px solid var(--rl-amber-dark) !important; border-radius: var(--rl-radius) !important; background: #080a0b !important; }
[data-testid="stMetricLabel"] p { color: #777f82 !important; font: 700 .65rem/1.25 "SFMono-Regular",Consolas,monospace !important; }
[data-testid="stMetricValue"] { color: #ecece6 !important; font-family: "SFMono-Regular",Consolas,monospace !important; font-variant-numeric: tabular-nums; }
[data-testid="stExpander"] { border: 1px solid var(--rl-line) !important; border-left: 3px solid #535a5e !important; border-radius: var(--rl-radius) !important; background: #080a0b !important; box-shadow: none !important; }
[data-testid="stExpander"] summary { font: 800 .66rem/1 "SFMono-Regular",Consolas,monospace !important; letter-spacing: .08em; color: #b5b8b5 !important; }
[data-testid="stAlert"] { border: 1px solid #353b3e !important; border-left: 3px solid var(--rl-amber-dark) !important; border-radius: var(--rl-radius) !important; background: #0c0e0f !important; color: #d7d8d2 !important; }

/* RAG workspace */
.rl-ai-shell {
    margin: .35rem 0 .8rem !important; padding: 1rem 1.05rem !important;
    border: 1px solid #393748 !important; border-top: 4px solid var(--rl-violet) !important;
    border-radius: var(--rl-radius) !important; background: #0a0a0d !important;
    background-image: none !important; box-shadow: none !important;
}
.rl-ai-shell::before { display: none !important; }
.rl-ai-kicker { color: var(--rl-amber) !important; font-size: .59rem !important; letter-spacing: .14em !important; }
.rl-ai-title { margin-top: .35rem !important; color: #ededeb !important; font-size: 1.02rem !important; letter-spacing: .01em !important; text-transform: uppercase; }
.rl-ai-copy { margin-top: .35rem !important; color: #93999c !important; font-size: .76rem !important; line-height: 1.5 !important; }
.rl-chip { padding: .28rem .42rem !important; border: 1px solid #383646 !important; border-radius: 2px !important; background: #0d0c12 !important; color: #aaa6c1 !important; font-size: .56rem !important; }
.rl-ai-report { border-radius: var(--rl-radius) !important; background: #08090b !important; box-shadow: none !important; }
.rl-ai-report-header, .rl-ai-block { border-color: var(--rl-line) !important; background: transparent !important; }
.rl-ai-block-label { color: var(--rl-amber) !important; }
.rl-ai-list-item, .rl-evidence-card { border-radius: var(--rl-radius) !important; background: #0a0c0e !important; box-shadow: none !important; }
.rl-evidence-card { border-left: 3px solid var(--rl-violet) !important; }

.rl-mode-note { border: 1px solid #5b4b27 !important; border-left: 3px solid var(--rl-amber) !important; border-radius: var(--rl-radius) !important; background: #100d06 !important; color: var(--rl-amber-bright) !important; font-size: .61rem !important; }
.rl-footer { margin-top: 1.3rem !important; padding-top: .75rem !important; border-top: 1px solid var(--rl-line) !important; color: #626a6d !important; font-size: .59rem !important; }

@media (max-width: 1100px) {
    .block-container { padding-left: 1.4rem !important; padding-right: 1.4rem !important; }
    .rl-runtime-grid { grid-template-columns: repeat(2,minmax(0,1fr)); }
    .rl-runtime-card:nth-child(2) { border-right: 0 !important; }
    .rl-runtime-card:nth-child(-n+2) { border-bottom: 1px solid var(--rl-line) !important; }
}
@media (max-width: 760px) {
    .block-container { padding: 1.2rem .85rem 2rem !important; }
    .rl-runtime-grid { grid-template-columns: 1fr; }
    .rl-runtime-card { border-right: 0 !important; border-bottom: 1px solid var(--rl-line) !important; }
    .rl-countdown { align-items: flex-start !important; flex-direction: column !important; }
    .rl-service-grid { grid-template-columns: repeat(2,minmax(0,1fr)) !important; }
}
</style>
"""


def apply_console_styles() -> None:
    """Apply the centralized RootLensAI console design system."""
    st.markdown(CONSOLE_CSS, unsafe_allow_html=True)
