"""Model confidence and calibration presentation for RootLensAI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CALIBRATION_METRICS = (
    REPO_ROOT
    / "rootlens/data/reports/graphsage_rca_v2_calibration/"
    / "validation_calibration_metrics.json"
)


@st.cache_data(ttl=30)
def load_calibration_metrics(
    path: str = str(DEFAULT_CALIBRATION_METRICS),
) -> dict[str, Any] | None:
    """Load offline calibration metrics produced by evaluate_calibration_v2.py."""

    metrics_path = Path(path)

    if not metrics_path.is_file():
        return None

    try:
        payload = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None

    return payload


def _confidence_interpretation(
    confidence: float,
) -> tuple[str, str]:
    """
    Provide UI-only interpretation of class separation.

    These bands are presentation aids, not statistical guarantees.
    """

    if confidence >= 0.90:
        return (
            "Strong class separation",
            (
                "The predicted class clearly dominates the current "
                "probability distribution."
            ),
        )

    if confidence >= 0.70:
        return (
            "Moderate class separation",
            (
                "The model favors this class, although competing "
                "root-cause classes retain meaningful probability."
            ),
        )

    return (
        "Low class separation",
        (
            "Competing RCA classes retain substantial probability. "
            "Treat this diagnosis as uncertain and inspect the full "
            "probability distribution."
        ),
    )


def render_calibration_panel(
    result: dict[str, Any],
) -> None:
    """
    Render current prediction confidence separately from offline
    model calibration evidence.
    """

    confidence = float(
        result.get("confidence", 0.0)
    )

    predicted = str(
        result.get(
            "predicted_root_cause",
            "unknown",
        )
    )

    probabilities = result.get(
        "probabilities",
        {},
    )

    interpretation, interpretation_copy = (
        _confidence_interpretation(
            confidence
        )
    )

    remaining_probability = max(
        0.0,
        1.0 - confidence,
    )

    st.markdown(
        '<div class="rl-section-label">'
        'CONFIDENCE · MODEL RELIABILITY'
        '</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Current live-window confidence
    # ------------------------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### Current telemetry-window confidence"
        )

        current_left, current_middle, current_right = st.columns(
            [1.0, 1.15, 2.4],
            gap="large",
        )

        with current_left:
            st.metric(
                "Predicted class",
                predicted.replace(
                    "_",
                    " ",
                ).upper(),
            )

        with current_middle:
            st.metric(
                "Model confidence",
                f"{confidence * 100:.1f}%",
            )

        with current_right:
            st.markdown(
                f"**{interpretation}**"
            )

            st.write(
                interpretation_copy
            )

        st.caption(
            "Confidence is the probability assigned to the predicted "
            "class for this telemetry window. It is not the model's "
            "overall accuracy."
        )

        if confidence < 0.90:
            st.caption(
                f"{remaining_probability * 100:.1f}% of the current "
                "probability mass remains assigned to competing classes."
            )

        if probabilities:
            ordered = sorted(
                probabilities.items(),
                key=lambda item: float(
                    item[1]
                ),
                reverse=True,
            )

            if len(ordered) >= 2:
                first_name, first_prob = ordered[0]
                second_name, second_prob = ordered[1]

                margin = float(
                    first_prob
                ) - float(
                    second_prob
                )

                st.caption(
                    "Top-class margin: "
                    f"{margin * 100:.1f} percentage points "
                    f"({first_name} vs {second_name})."
                )

    # ------------------------------------------------------------------
    # Offline calibration evidence
    # ------------------------------------------------------------------

    calibration = load_calibration_metrics()

    with st.expander(
        "OFFLINE MODEL RELIABILITY METRICS · DATASET V2 VALIDATION",
        expanded=False,
    ):

        if calibration is None:
            st.warning(
                "Calibration artifact was not found. Run "
                "`python rootlens/scripts/evaluate_calibration_v2.py` "
                "to generate the validation calibration report."
            )
            return

        accuracy = float(
            calibration.get(
                "validation_accuracy",
                0.0,
            )
        )

        mean_confidence = float(
            calibration.get(
                "validation_mean_confidence",
                0.0,
            )
        )

        ece = float(
            calibration.get(
                "validation_ece",
                0.0,
            )
        )

        confidence_gap = float(
            calibration.get(
                "validation_accuracy_confidence_gap",
                0.0,
            )
        )

        brier = float(
            calibration.get(
                "validation_brier_score",
                0.0,
            )
        )

        nll = float(
            calibration.get(
                "validation_nll",
                0.0,
            )
        )

        metric_1, metric_2, metric_3, metric_4 = st.columns(
            4,
            gap="medium",
        )

        metric_1.metric(
            "Validation accuracy",
            f"{accuracy * 100:.2f}%",
        )

        metric_2.metric(
            "Mean confidence",
            f"{mean_confidence * 100:.2f}%",
        )

        metric_3.metric(
            "Expected Calibration Error",
            f"{ece * 100:.2f}%",
        )

        metric_4.metric(
            "Accuracy-confidence gap",
            f"{confidence_gap * 100:.2f} pp",
        )

        st.divider()

        score_left, score_right = st.columns(
            2,
            gap="large",
        )

        with score_left:
            st.metric(
                "Multiclass Brier score",
                f"{brier:.4f}",
            )

            st.caption(
                "Measures the quality of the complete class-probability "
                "distribution. Lower is better."
            )

        with score_right:
            st.metric(
                "Negative Log-Likelihood",
                f"{nll:.4f}",
            )

            st.caption(
                "Penalizes confidently incorrect predictions. "
                "Lower is better."
            )

        st.info(
            "Expected Calibration Error measures how closely the "
            "model's reported confidence matches observed correctness "
            "across validation confidence ranges. RootLens calibration "
            "is evaluated offline; it is not recomputed from live "
            "telemetry."
        )

        st.caption(
            "Dataset v2 validation only · raw softmax probabilities · "
            "sealed test set not evaluated."
        )

        st.caption(
            "Confidence interpretation bands shown in the live UI are "
            "presentation aids for class separation, not statistical "
            "guarantees."
        )
