"""
scorer.py – Combine ML model predictions with hard-gating rules and sub-score estimation.

Hard gates protect against obvious mis-scoring by capping sub-scores when
specific feature thresholds are violated.  Gates **never** inflate a score;
they only impose upper bounds.
"""

from __future__ import annotations

import numpy as np
from scorer.confidence import calculate_confidence
from scorer.diagnostic_engine import analyze as analyze_diagnostic

# Band-score clipping range.
_BAND_MIN: float = 4.0
_BAND_MAX: float = 9.0


# ---------------------------------------------------------------------------
# IELTS band rounding
# ---------------------------------------------------------------------------

def round_to_ielts_band(score: float) -> float:
    """Round *score* to the nearest 0.5 step within [4.0, 9.0].

    Examples
    --------
    >>> round_to_ielts_band(6.3)
    6.5
    >>> round_to_ielts_band(3.2)
    4.0
    """
    clamped = max(_BAND_MIN, min(_BAND_MAX, score))
    return round(clamped * 2) / 2


# ---------------------------------------------------------------------------
# Hard gating
# ---------------------------------------------------------------------------

def apply_hard_gates(
    ml_scores: dict[str, float],
    norm_features: dict[str, float],
    raw_features: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    """Apply deterministic hard gates on top of ML predictions.

    Parameters
    ----------
    ml_scores:
        Sub-scores keyed by criterion
        (``task_response``, ``lexical``, ``grammar``, ``coherence``).
    norm_features:
        Normalised features (all in [0, 1]).
    raw_features:
        Original un-normalised feature values.

    Returns
    -------
    tuple[dict[str, float], list[str]]
        * Gated scores (clipped to [4.0, 9.0]).
        * List of gate names that fired (for explainability logs).
    """
    scores = {k: float(v) for k, v in ml_scores.items()}
    fired: list[str] = []

    # Gate 1 – very low topic similarity -> task_response floor at 4.0
    sim_raw = raw_features.get("topic_similarity", 1.0)
    if sim_raw < 0.35:
        scores["task_response"] = _BAND_MIN  # hard floor
        fired.append(f"gate:low_topic_similarity_raw({sim_raw:.3f}<0.35)->TR_floor_4.0")

    # Gate 2 – low task coverage -> task_response capped at 5.0
    task_cov_raw = raw_features.get("task_coverage_score", 1.0)
    if task_cov_raw < 0.35:
        scores["task_response"] = min(scores.get("task_response", _BAND_MAX), 5.0)
        fired.append(f"gate:low_task_coverage_raw({task_cov_raw:.2f}<0.35)->TR_cap_5.0")

    # Gate 3 – very high grammar error rate -> grammar capped at 5.0
    grammar_err_raw = raw_features.get("grammar_error_rate", 0.0)
    if grammar_err_raw > 4.5:
        scores["grammar"] = min(scores.get("grammar", _BAND_MAX), 5.0)
        fired.append(f"gate:high_grammar_errors_raw({grammar_err_raw:.2f}>4.5)->GRA_cap_5.0")

    # Clip everything to valid band range.
    scores = {k: float(np.clip(v, _BAND_MIN, _BAND_MAX)) for k, v in scores.items()}

    return scores, fired


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score_essay(
    normalized_features: dict[str, float],
    raw_features: dict[str, float],
    ml_predictions: dict[str, float],
) -> dict[str, object]:
    """Score an essay by merging ML predictions with hard gates.

    Parameters
    ----------
    normalized_features:
        Feature values normalised to [0, 1].
    raw_features:
        Original (un-normalised) feature values — needed by gates that
        reference absolute thresholds.
    ml_predictions:
        Band-score predictions from :class:`ml_model.IELTSModel`.

    Returns
    -------
    dict
        ``scores``        – gated sub-scores and overall (float).
        ``bands``         – rounded IELTS bands (float, step 0.5).
        ``gates``         – list of gate names that fired.
        ``confidence``    – confidence score [0.0, 1.0].
        ``is_estimated``  – True if sub-scores are estimated.
    """
    # 1. Determine modes and prepare base sub-scores
    sub_keys = ["task_response", "lexical", "grammar", "coherence"]
    has_subscores = all(k in ml_predictions for k in sub_keys)
    has_overall = "overall" in ml_predictions

    if has_overall and not has_subscores:
        # Mode A: Overall-only model -> estimate subscores from feature groups
        overall_raw = ml_predictions["overall"]
        # Run estimation anchor to overall
        est_res = analyze_diagnostic(normalized_features, raw_features, overall_raw, is_estimated=True)
        sub_scores = est_res["sub_scores"]
        is_estimated = True
    elif has_subscores:
        # Mode B or Backward Compatibility: ML predicted subscores
        sub_scores = {k: ml_predictions[k] for k in sub_keys}
        overall_raw = ml_predictions.get("overall", sum(sub_scores.values()) / 4.0)
        is_estimated = False
    else:
        # Fallback if ml_predictions is somehow empty
        sub_scores = {k: 6.0 for k in sub_keys}
        overall_raw = 6.0
        is_estimated = True

    # 2. Apply hard gates to the sub-scores
    gated_subscores, gates = apply_hard_gates(
        ml_scores=sub_scores,
        norm_features=normalized_features,
        raw_features=raw_features,
    )

    # 3. Apply proportional reduction to the overall score
    reduction = sum(sub_scores[k] - gated_subscores[k] for k in sub_keys) / 4.0
    overall_gated = max(_BAND_MIN, min(_BAND_MAX, overall_raw - reduction))

    # Construct final gated scores dict
    scores = dict(gated_subscores)
    scores["overall"] = float(overall_gated)

    bands = {k: round_to_ielts_band(v) for k, v in scores.items()}

    # 4. Calculate prediction confidence
    confidence = calculate_confidence(raw_features, normalized_features, overall_gated)

    return {
        "scores": scores,
        "bands": bands,
        "gates": gates,
        "confidence": confidence,
        "is_estimated": is_estimated,
    }
