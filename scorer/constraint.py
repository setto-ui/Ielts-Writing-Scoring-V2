"""
constraint.py – IELTS ceiling rules applied AFTER scoring, BEFORE calibration.

These constraints encode examiner heuristics that enforce cross-criterion
consistency.  For example, a weak Task Response should prevent unrealistically
high Coherence or Lexical Resource scores.

Every constraint only *caps* (lowers) scores — it never inflates them.
"""

from __future__ import annotations

# Band-score clipping range.
_BAND_MIN: float = 4.0
_BAND_MAX: float = 9.0


def apply_constraints(
    scores: dict[str, float],
    features: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    """Apply IELTS hard ceiling rules to *scores*.

    Parameters
    ----------
    scores:
        Sub-scores after ML prediction + hard gating.  Expected keys:
        ``task_response``, ``lexical``, ``grammar``, ``coherence``.
    features:
        Normalised (or raw, where noted) feature dict.  Used to look up
        ``topic_similarity`` and ``gaming_penalty``.

    Returns
    -------
    tuple[dict[str, float], list[str]]
        * ``constrained_scores`` – adjusted sub-scores, clipped to [4.0, 9.0].
        * ``applied_rules`` – human-readable list of constraint names that
          fired (useful for explainability / audit logs).
    """
    # Work on a mutable copy.
    s = {k: float(v) for k, v in scores.items()}
    applied: list[str] = []
    
    alpha = 0.8  # Penalty scaling factor

    # ---- Rule 1: weak Task Response penalises CC, LR, GRA ----
    tr_score = s.get("task_response", 9.0)
    if tr_score < 5.5:
        penalty = (5.5 - tr_score) * alpha
        for key in ("coherence", "lexical", "grammar"):
            s[key] -= penalty
            applied.append(f"penalty:TR_weak→{_short(key)}-{penalty:.2f}")

    # ---- Rule 2: weak Grammar penalises LR and CC ----
    gra_score = s.get("grammar", 9.0)
    if gra_score < 5.0:
        penalty = (5.0 - gra_score) * alpha
        for key in ("lexical", "coherence"):
            s[key] -= penalty
            applied.append(f"penalty:GRA_weak→{_short(key)}-{penalty:.2f}")

    # ---- Rule 3: weak Lexical Resource penalises TR ----
    lr_score = s.get("lexical", 9.0)
    if lr_score < 5.0:
        penalty = (5.0 - lr_score) * alpha
        s["task_response"] -= penalty
        applied.append(f"penalty:LR_weak→TR-{penalty:.2f}")

    # ---- Rule 4: off-topic → TR penalty ----
    topic_sim = features.get("topic_similarity", 1.0)
    if topic_sim < 0.40:
        # Instead of hard cap at 4.5, penalise proportionally if similarity is low
        penalty = (0.40 - topic_sim) * 10.0 * alpha
        s["task_response"] -= penalty
        applied.append(f"penalty:off_topic→TR-{penalty:.2f}")

    # ---- Rule 5: gaming penalty ----
    gaming_penalty = features.get("gaming_penalty", 0.0)
    if gaming_penalty > 0:
        tr_reduction = gaming_penalty * 0.5
        lr_reduction = gaming_penalty * 0.3
        s["task_response"] = s.get("task_response", _BAND_MIN) - tr_reduction
        s["lexical"] = s.get("lexical", _BAND_MIN) - lr_reduction
        applied.append(
            f"penalty:gaming({gaming_penalty:.2f})→TR-{tr_reduction:.2f},LR-{lr_reduction:.2f}"
        )

    # ---- Overall propagation ----
    if "overall" in s:
        sub_keys = ["task_response", "lexical", "grammar", "coherence"]
        initial_sub_sum = sum(float(scores[k]) for k in sub_keys)
        final_sub_sum = sum(s[k] for k in sub_keys)
        reduction = (initial_sub_sum - final_sub_sum) / 4.0
        s["overall"] -= reduction

    # ---- Final clipping ----
    s = {k: max(_BAND_MIN, min(_BAND_MAX, v)) for k, v in s.items()}

    return s, applied


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CRITERION_SHORT: dict[str, str] = {
    "task_response": "TR",
    "lexical": "LR",
    "grammar": "GRA",
    "coherence": "CC",
}


def _short(criterion: str) -> str:
    """Return the abbreviated name for a criterion."""
    return _CRITERION_SHORT.get(criterion, criterion.upper())
