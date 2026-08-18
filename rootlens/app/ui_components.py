"""Presentation-only components for the RootLensAI Streamlit console."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml


APP_CSS = """
<style>
  :root {
    --black: #000000;
    --panel: #090909;
    --panel-2: #0e0e0e;
    --white: #ffffff;
    --muted: #8b8b8b;
    --line: #292929;
    --line-soft: #191919;
    --incident: #ff3b30;
    --warning: #c5a84b;
  }
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: var(--black) !important;
    color: var(--white) !important;
  }
  html, body, [class*="css"], p, label, h1, h2, h3, span, div {
    font-family: Inter, "IBM Plex Sans", "Segoe UI", Arial, sans-serif;
    color: var(--white);
  }
  .block-container { max-width: 1240px; padding: 2.3rem 2rem 2rem; }
  [data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
  .rl-header { display:flex; justify-content:space-between; align-items:flex-end;
    gap:1.5rem; padding-bottom:1.35rem; border-bottom:1px solid var(--line); }
  .rl-brand { font-size:2.55rem; line-height:1; font-weight:760; letter-spacing:-0.055em; }
  .rl-subtitle { color:var(--muted)!important; font-size:.96rem; margin-top:.55rem; }
  .rl-badges { display:flex; justify-content:flex-end; gap:.5rem; flex-wrap:wrap; }
  .rl-badge { border:1px solid var(--line); background:var(--panel); padding:.43rem .62rem;
    font:600 .67rem/1.1 "IBM Plex Mono", "JetBrains Mono", Consolas, monospace;
    letter-spacing:.08em; color:#d4d4d4; white-space:nowrap; }
  .rl-badge b { color:var(--muted); font:inherit; margin-right:.32rem; }
  .rl-section-label { color:var(--muted)!important; font-size:.68rem; font-weight:700;
    letter-spacing:.14em; text-transform:uppercase; margin:1.7rem 0 .65rem; }
  .rl-status { display:grid; grid-template-columns:1.45fr 1fr .7fr; gap:0;
    border:1px solid var(--line); background:var(--panel); margin:1rem 0 1.6rem; }
  .rl-status > div { min-height:132px; padding:1.25rem 1.35rem; display:flex;
    flex-direction:column; justify-content:center; border-right:1px solid var(--line); }
  .rl-status > div:last-child { border-right:0; }
  .rl-status-state { font-size:1.8rem; font-weight:760; letter-spacing:-.025em; }
  .rl-status.incident { border-color:#65201c; }
  .rl-status.incident .rl-status-state, .rl-status.incident .rl-root-value,
  .rl-incident-text { color:var(--incident)!important; }
  .rl-field-label { color:var(--muted)!important; font-size:.65rem; font-weight:700;
    letter-spacing:.12em; text-transform:uppercase; margin-bottom:.48rem; }
  .rl-root-value { font-size:2rem; font-weight:760; text-transform:uppercase;
    letter-spacing:-.025em; }
  .rl-confidence { font:700 2rem/1 "IBM Plex Mono", "JetBrains Mono", Consolas, monospace; }
  .rl-service-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr));
    border-top:1px solid var(--line); border-left:1px solid var(--line); }
  .rl-service { display:flex; justify-content:space-between; gap:.7rem; padding:.7rem .8rem;
    background:var(--panel); border-right:1px solid var(--line); border-bottom:1px solid var(--line); }
  .rl-service-name { font:500 .76rem/1.2 "IBM Plex Mono", Consolas, monospace; }
  .rl-service-state { color:var(--muted)!important; font:700 .62rem/1.2 "IBM Plex Mono", Consolas, monospace;
    letter-spacing:.08em; }
  .rl-service.faulty { background:#160605; border-color:#65201c; }
  .rl-service.faulty .rl-service-name, .rl-service.faulty .rl-service-state { color:var(--incident)!important; }
  .rl-inference-note { color:var(--muted)!important; font-size:.72rem; margin-top:.55rem; }
  .rl-prob-row { display:grid; grid-template-columns:150px 1fr 62px; gap:.8rem;
    align-items:center; padding:.42rem 0; }
  .rl-prob-label, .rl-prob-value { font:500 .75rem/1 "IBM Plex Mono", Consolas, monospace; }
  .rl-prob-value { text-align:right; color:#d2d2d2; }
  .rl-track { height:7px; background:#181818; overflow:hidden; }
  .rl-fill { height:100%; background:#bdbdbd; min-width:1px; }
  .rl-prob-row.predicted .rl-fill { background:#ffffff; }
  .rl-prob-row.incident .rl-fill { background:var(--incident); }
  .rl-prob-row.incident .rl-prob-label, .rl-prob-row.incident .rl-prob-value { color:var(--incident)!important; }
  .rl-mode-note { display:inline-block; border:1px solid #4a4124; color:#cdbb76!important;
    background:#100e06; padding:.35rem .55rem; margin:.25rem 0 .5rem;
    font:600 .68rem/1 "IBM Plex Mono", Consolas, monospace; letter-spacing:.06em; }
  .rl-separation-note { border-left:2px solid #777; background:var(--panel); padding:.75rem .9rem;
    color:var(--muted)!important; font-size:.75rem; margin-bottom:.8rem; }
  .rl-separation-note b { color:#fff; font:700 .68rem/1 "IBM Plex Mono", Consolas, monospace;
    letter-spacing:.1em; margin-right:.45rem; }
  .rl-injection-state { display:flex; align-items:center; justify-content:space-between; gap:1rem;
    border:1px solid var(--line); background:var(--panel); padding:.72rem .85rem; margin:.75rem 0; }
  .rl-injection-state span { color:var(--muted)!important; font:700 .64rem/1 "IBM Plex Mono", Consolas, monospace;
    letter-spacing:.11em; }
  .rl-injection-state b { font:700 .75rem/1 "IBM Plex Mono", Consolas, monospace; }
  .rl-injection-state.active { border-color:#65201c; background:#160605; }
  .rl-injection-state.active b { color:var(--incident)!important; }
  .rl-footer { border-top:1px solid var(--line); margin-top:2rem; padding-top:.8rem;
    color:var(--muted)!important; font:500 .64rem/1.5 "IBM Plex Mono", Consolas, monospace;
    letter-spacing:.065em; }
  div[data-testid="stDataFrame"] { border:1px solid var(--line); background:var(--panel); }
  div[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); padding:1rem; }
  div[data-testid="stExpander"] { border:1px solid var(--line); background:var(--panel); border-radius:0; }
  .stButton > button { background:#ffffff; color:#000000; border:1px solid #ffffff;
    border-radius:0; font-weight:750; min-height:2.55rem; }
  .stButton > button:hover { background:#d8d8d8; color:#000000; border-color:#d8d8d8; }
  [data-testid="stToggle"] label span { color:#d0d0d0!important; }
  hr { border-color:var(--line)!important; }
  code { font-family:"IBM Plex Mono", "JetBrains Mono", Consolas, monospace!important; }
  @media (max-width: 820px) {
    .block-container { padding:1.5rem 1rem; }
    .rl-header { align-items:flex-start; flex-direction:column; }
    .rl-badges { justify-content:flex-start; }
    .rl-status { grid-template-columns:1fr; }
    .rl-status > div { min-height:95px; border-right:0; border-bottom:1px solid var(--line); }
    .rl-status > div:last-child { border-bottom:0; }
    .rl-service-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .rl-prob-row { grid-template-columns:110px 1fr 56px; }
  }
</style>
"""


FIXED_POSITIONS = {
    "frontend-proxy": (45, 215),
    "frontend": (180, 215),
    "ad": (360, 35),
    "currency": (360, 95),
    "cart": (360, 155),
    "checkout": (360, 215),
    "shipping": (360, 275),
    "recommendation": (360, 335),
    "product-catalog": (590, 155),
    "payment": (590, 215),
    "email": (590, 275),
    "quote": (590, 335),
}


def header_html(demo_mode: bool) -> str:
    mode = "DEMO" if demo_mode else "LIVE"
    return f"""
    <div class="rl-header">
      <div><div class="rl-brand">RootLensAI</div>
      <div class="rl-subtitle">AI-Driven Microservice Root Cause Analysis</div></div>
      <div class="rl-badges">
        <div class="rl-badge"><b>MODEL</b>GraphSAGE RCA v2</div>
        <div class="rl-badge"><b>MODE</b>{mode}</div>
        <div class="rl-badge"><b>GRAPH</b>12 SERVICES</div>
      </div>
    </div>"""


def status_html(result: Mapping[str, object]) -> str:
    incident = result["system_status"] == "incident"
    predicted = str(result["predicted_root_cause"]).replace("_", " ")
    return f"""
    <div class="rl-status {'incident' if incident else ''}">
      <div><div class="rl-field-label">System status</div>
        <div class="rl-status-state">{'INCIDENT DETECTED' if incident else 'SYSTEM HEALTHY'}</div></div>
      <div><div class="rl-field-label">{'Predicted root cause' if incident else 'Predicted state'}</div>
        <div class="rl-root-value">{html.escape(predicted)}</div></div>
      <div><div class="rl-field-label">Confidence</div>
        <div class="rl-confidence">{float(result['confidence']):.1%}</div></div>
    </div>"""


def service_status_html(services: list[str], predicted: str) -> str:
    faulty = predicted.replace("_", "-") if predicted != "healthy" else None
    items = []
    for service in services:
        is_faulty = service == faulty
        items.append(
            f'<div class="rl-service {"faulty" if is_faulty else ""}">'
            f'<span class="rl-service-name">{html.escape(service)}</span>'
            f'<span class="rl-service-state">{"MODEL RCA" if is_faulty else "NEUTRAL"}</span></div>'
        )
    return (
        '<div class="rl-service-grid">' + "".join(items) + "</div>"
        '<div class="rl-inference-note">Status reflects model-indicated root cause only; '
        'neutral labels are not direct service health probes.</div>'
    )


def probabilities_html(probabilities: Mapping[str, float], predicted: str) -> str:
    rows = []
    for label, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
        is_predicted = label == predicted
        is_incident = is_predicted and predicted != "healthy"
        css = "incident" if is_incident else "predicted" if is_predicted else ""
        rows.append(
            f'<div class="rl-prob-row {css}"><div class="rl-prob-label">{html.escape(label)}</div>'
            f'<div class="rl-track"><div class="rl-fill" style="width:{float(probability)*100:.4f}%"></div></div>'
            f'<div class="rl-prob-value">{float(probability):.1%}</div></div>'
        )
    return "".join(rows)


def topology_svg(topology_path: Path, predicted: str) -> str:
    topology = yaml.safe_load(topology_path.read_text(encoding="utf-8"))
    nodes = list(topology["node_order"])
    if set(nodes) != set(FIXED_POSITIONS):
        raise ValueError("Frozen topology nodes do not match the fixed MVP layout")
    root = predicted.replace("_", "-") if predicted != "healthy" else None
    edge_parts = []
    for edge in topology["edges"]:
        x1, y1 = FIXED_POSITIONS[edge["source"]]
        x2, y2 = FIXED_POSITIONS[edge["target"]]
        edge_parts.append(
            f'<line x1="{x1+55}" y1="{y1+15}" x2="{x2+55}" y2="{y2+15}" '
            'stroke="#3b3b3b" stroke-width="1" marker-end="url(#arrow)" />'
        )
    node_parts = []
    for node in nodes:
        x, y = FIXED_POSITIONS[node]
        active = node == root
        stroke = "#ff3b30" if active else "#6a6a6a"
        fill = "#180605" if active else "#090909"
        text = "#ff3b30" if active else "#ffffff"
        node_parts.append(
            f'<rect x="{x}" y="{y}" width="110" height="30" rx="2" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{2 if active else 1}" />'
            f'<text x="{x+55}" y="{y+19}" text-anchor="middle" fill="{text}" '
            'font-family="IBM Plex Mono,Consolas,monospace" font-size="10">'
            f'{html.escape(node)}</text>'
        )
    return f"""
    <div style="border:1px solid #292929;background:#050505;padding:.55rem;overflow-x:auto">
    <svg viewBox="0 0 750 390" width="100%" role="img" aria-label="RootLens service topology">
      <defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
        <path d="M0,0 L7,3.5 L0,7 z" fill="#3b3b3b" /></marker></defs>
      {''.join(edge_parts)}{''.join(node_parts)}
    </svg></div>"""


def telemetry_style(frame: pd.DataFrame, predicted: str) -> pd.io.formats.style.Styler:
    target = predicted.replace("_", "-") if predicted != "healthy" else None

    def row_style(row: pd.Series) -> list[str]:
        if target and row["service"] == target:
            return ["background-color:#1b0706;color:#ff6b63"] * len(row)
        return ["background-color:#090909;color:#f5f5f5"] * len(row)

    return frame.style.apply(row_style, axis=1).format({
        "cpu": "{:.2f}", "memory": "{:.2f}", "request_rate": "{:.3f}",
        "latency_ms": "{:.2f}", "error_rps": "{:.4f}", "error_rate": "{:.4f}",
    })
