"""Runtime state, timers, fault controls, and shared UI helpers."""

from __future__ import annotations

import html
import logging
import math
import time
import textwrap

import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

from rootlens.demo.fault_controller import (
    FaultControllerError,
    get_fault_state,
    inject_fault,
    restore_fault,
    supported_faults,
)

LOGGER = logging.getLogger("rootlens.mvp")

INJECT_RCA_LOCK_SECONDS = 30
RESTORE_RCA_LOCK_SECONDS = 60
COUNTDOWN_REFRESH_MS = 1000


def render_html(markup: str) -> None:
    """Render indented HTML safely without Markdown turning it into a code block."""
    st.markdown(
        textwrap.dedent(markup).strip(),
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    render_html(
        f"""
        <div class="rl-section-label">{html.escape(title)}</div>
        """
    )


def format_seconds(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 1:
        return f"{value:.2f}s"
    return f"{value:.1f}s"


def init_ui_state() -> None:
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
    unlock_at = float(st.session_state.get("rca_unlock_at", 0.0) or 0.0)
    return int(math.ceil(max(0.0, unlock_at - time.time())))


def cooldown_active() -> bool:
    return cooldown_remaining() > 0


def start_rca_cooldown(seconds: int, reason: str) -> None:
    st.session_state.rca_unlock_at = time.time() + seconds
    st.session_state.rca_lock_reason = reason
    st.session_state.rca_lock_seconds = seconds
    st.session_state.rca_stale = True


def maybe_tick_countdown() -> None:
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

    # NOTE:
    # Using render_html() / textwrap.dedent() is important here.
    # Indented HTML separated by blank lines can otherwise be interpreted
    # by Markdown as a code block, which is what caused visible <div> tags.
    render_html(
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
        """
    )


def render_cooldown_banner() -> None:
    remaining = cooldown_remaining()

    if remaining:
        reason = st.session_state.get("rca_lock_reason", "Telemetry stabilizing")
        total = st.session_state.get("rca_lock_seconds", remaining)

        render_html(
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
            """
        )

    elif st.session_state.get("last_operation_type"):
        render_html(
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
            """
        )


def render_stale_rca_banner() -> None:
    if not st.session_state.get("rca_stale"):
        return

    remaining = cooldown_remaining()
    suffix = (
        f" New inference unlocks in {remaining}s."
        if remaining
        else " The wait period has elapsed; refresh RCA for the current telemetry window."
    )

    render_html(
        f"""
        <div class="rl-stale-banner">
          <b>RCA RESULT INVALIDATED</b><br>
          <span>
            The testbed changed after the previous inference.{html.escape(suffix)}
            The previous result is hidden to avoid presenting stale telemetry
            as the current system state.
          </span>
        </div>
        """
    )


def render_demo_controls() -> None:
    section("Demo controls · fault injection")

    render_html(
        """
        <div class="rl-separation-note">
          <b>FAULT INJECTION</b>
          changes the testbed only. Injected fault metadata is not passed
          to the RCA model. RCA refresh is temporarily locked after a
          testbed change so the next 30-second PromQL window is interpretable.
        </div>
        """
    )

    choices = supported_faults()
    control_left, control_right = st.columns(2, gap="medium")

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

    render_html(
        f"""
        <div class="rl-injection-state {state_class}">
          <span>INJECTION STATE</span>
          <b>{html.escape(state_text)}</b>
        </div>
        """
    )

    last_operation = st.session_state.get("last_operation_type")
    last_duration = st.session_state.get("last_operation_duration")

    if last_operation and last_duration is not None:
        render_html(
            f"""
            <div class="rl-operation-result">
              <span class="ok">●</span>
              <span>
                <strong>{html.escape(str(last_operation))}</strong>
                completed in <strong>{format_seconds(last_duration)}</strong>
              </span>
            </div>
            """
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

    with inject_column.container(key="fault_inject_action"):
        inject_clicked = st.button(
            "INJECT FAULT",
            width="stretch",
            disabled=inject_disabled,
        )

    if inject_clicked:
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
            st.error(f"Fault injection failed after {duration:.2f}s: {exc}")

    with restore_column.container(key="restore_system_action"):
        restore_clicked = st.button(
            "RESTORE SYSTEM",
            width="stretch",
            disabled=restore_disabled,
        )

    if restore_clicked:
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
            st.error(f"System restoration failed after {duration:.2f}s: {exc}")
