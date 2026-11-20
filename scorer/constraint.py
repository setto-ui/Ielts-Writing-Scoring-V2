"""
constraint.py – IELTS ceiling rules applied AFTER scoring, BEFORE calibration.

These constraints encode examiner heuristics that enforce cross-criterion
consistency.  For example, a weak Task Response should prevent unrealistically
high Coherence or Lexical Resource scores.

Every constraint only *caps* (lowers) scores — it never inflates them.
"""

from __future__ import annotations
import numpy as np

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

    # ---- Rule 4: off-topic / tangential response → TR penalty ----
    topic_sim = features.get("topic_similarity", 1.0)
    is_tangential = features.get("is_tangential_drift", 0.0)
    if is_tangential > 0.5:
        # Confirmed tangential / off-topic response in IELTS (TR capped at 4.5, CC capped at 5.5)
        s["task_response"] = min(s.get("task_response", _BAND_MAX), 4.5)
        s["coherence"] = min(s.get("coherence", _BAND_MAX), 5.5)
        applied.append("penalty:tangential_topic_drift→TR_cap_4.5,CC_cap_5.5")
    elif topic_sim < 0.60:
        penalty = (0.60 - topic_sim) * 10.0 * alpha
        s["task_response"] -= penalty
        applied.append(f"penalty:low_topic_sim({topic_sim:.3f}<0.60)→TR-{penalty:.2f}")

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

    # ---- Rule 6: Lexical Register Ceiling (IELTS Band 5/6 descriptors) ----
    r_rare = features.get("rare_word_ratio", 0.0)
    r_awl = features.get("awl_ratio", 0.0)
    r_acad = features.get("academic_word_ratio", 0.0)
    if r_rare < 0.035 and (r_awl < 0.015 or r_acad < 0.015):
        # Purely elementary/childish vocabulary (Band 5.0 ceiling for LR)
        s["lexical"] = min(s.get("lexical", _BAND_MAX), 5.0)
        applied.append(f"ceiling:elementary_lexis(rare={r_rare:.3f},awl={r_awl:.3f})→LR_cap_5.0")
    elif r_rare < 0.060 and (r_awl < 0.020 or r_acad < 0.020):
        # Limited vocabulary range (Band 5.5 ceiling for LR)
        s["lexical"] = min(s.get("lexical", _BAND_MAX), 5.5)
        applied.append(f"ceiling:limited_lexis(rare={r_rare:.3f},awl={r_awl:.3f})→LR_cap_5.5")

    # ---- Rule 7: Grammatical Range Ceiling (IELTS Band 5/6 descriptors) ----
    sent_var = features.get("sent_length_variance", 0.0)
    avg_sl = features.get("avg_sentence_length", 0.0)
    complex_cr = features.get("complex_clause_ratio", 0.0)
    if sent_var < 35.0 and avg_sl < 16.0 and complex_cr < 1.0:
        # Monotonous simple sentence structures without variation (Band 5.0 ceiling for GRA)
        s["grammar"] = min(s.get("grammar", _BAND_MAX), 5.0)
        applied.append(f"ceiling:simple_syntax_range(var={sent_var:.1f},len={avg_sl:.1f})→GRA_cap_5.0")
    elif sent_var < 45.0 and avg_sl < 18.0 and complex_cr < 1.1:
        # Limited sentence range (Band 5.5 ceiling for GRA)
        s["grammar"] = min(s.get("grammar", _BAND_MAX), 5.5)
        applied.append(f"ceiling:limited_syntax_range(var={sent_var:.1f},len={avg_sl:.1f})→GRA_cap_5.5")

    # ---- Rule 8: Task Response Argument Depth Ceiling ----
    arg_depth = features.get("argument_depth_score", 0.0)
    if arg_depth < 0.20:
        # Minimal argument development / superficial claims
        s["task_response"] = min(s.get("task_response", _BAND_MAX), 5.0)
        applied.append(f"ceiling:minimal_argument_depth({arg_depth:.2f}<0.20)→TR_cap_5.0")
    elif arg_depth < 0.25:
        s["task_response"] = min(s.get("task_response", _BAND_MAX), 5.5)
        applied.append(f"ceiling:shallow_argument_depth({arg_depth:.2f}<0.25)→TR_cap_5.5")

    # ---- Overall propagation ----
    if "overall" in s:
        sub_keys = ["task_response", "lexical", "grammar", "coherence"]
        initial_sub_sum = sum(float(scores[k]) for k in sub_keys)
        final_sub_sum = sum(s[k] for k in sub_keys)
        reduction = (initial_sub_sum - final_sub_sum) / 4.0
        
        # Cambridge IELTS official rule: overall must be anchored to mean of 4 criteria
        cambridge_sub_mean = float(np.mean([s[k] for k in sub_keys]))
        s["overall"] = max(_BAND_MIN, min(s["overall"] - reduction, cambridge_sub_mean + 0.25))

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
