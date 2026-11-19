"""
logger.py
Pipeline state logging for debugging and tuning.
Logs full feature → score transformation at every stage.
"""

from datetime import datetime, timezone


def build_pipeline_log(
    raw_features: dict,
    normalized_features: dict,
    ml_predictions: dict,
    post_gate_scores: dict,
    post_constraint_scores: dict,
    final_band: float,
    applied_constraints: list,
    gate_decisions: list | None = None,
) -> dict:
    """
    Build a full pipeline state log for debugging.

    Args:
        raw_features: merged raw feature dict from all extractors
        normalized_features: after normalizer
        ml_predictions: raw ML model output
        post_gate_scores: after hard gating in scorer.py
        post_constraint_scores: after constraint.py ceiling rules
        final_band: calibrated overall band
        applied_constraints: list of constraint names that fired
        gate_decisions: list of gate names that fired (from scorer.py)

    Returns:
        dict with full pipeline state, deltas, and timestamp
    """
    # Compute deltas between stages
    gate_vs_ml = {}
    for k in ml_predictions:
        gate_val = post_gate_scores.get(k, ml_predictions[k])
        gate_vs_ml[k] = round(gate_val - ml_predictions[k], 3)

    constraint_vs_gate = {}
    for k in post_gate_scores:
        constraint_val = post_constraint_scores.get(k, post_gate_scores[k])
        constraint_vs_gate[k] = round(constraint_val - post_gate_scores[k], 3)

    # Serialise features (make sure floats are rounded for readability)
    def _round_dict(d: dict, decimals: int = 4) -> dict:
        result = {}
        for k, v in d.items():
            if isinstance(v, float):
                result[k] = round(v, decimals)
            elif isinstance(v, bool):
                result[k] = v
            elif isinstance(v, (list, dict)):
                result[k] = v  # keep as-is
            else:
                result[k] = v
        return result

    # Extract key raw gating features for quick inspection
    gating_snapshot = {
        "topic_similarity_raw": raw_features.get("topic_similarity"),
        "task_coverage_score_raw": raw_features.get("task_coverage_score"),
        "grammar_error_rate_raw": raw_features.get("grammar_error_rate"),
        "gaming_penalty_raw": raw_features.get("gaming_penalty"),
    }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline": {
            "1_raw_features": _round_dict(raw_features),
            "2_normalized": _round_dict(normalized_features),
            "3_ml_predictions": _round_dict(ml_predictions),
            "4_post_gate": _round_dict(post_gate_scores),
            "5_post_constraint": _round_dict(post_constraint_scores),
            "6_final_band": final_band,
        },
        "gating": {
            "raw_values_used": _round_dict(gating_snapshot),
            "gate_decisions": gate_decisions or [],
        },
        "deltas": {
            "gate_vs_ml": gate_vs_ml,
            "constraint_vs_gate": constraint_vs_gate,
        },
        "constraints_applied": applied_constraints,
    }


def format_log_summary(log: dict) -> str:
    """Return a compact human-readable summary of a pipeline log entry."""
    lines = []
    p = log.get("pipeline", {})
    ml = p.get("3_ml_predictions", {})
    gate = p.get("4_post_gate", {})
    con = p.get("5_post_constraint", {})
    band = p.get("6_final_band", "?")

    lines.append(f"Final Band: {band}")
    lines.append(f"  ML preds:     {ml}")
    lines.append(f"  Post-gate:    {gate}")
    lines.append(f"  Post-constr:  {con}")

    gating_info = log.get("gating", {})
    raw_vals = gating_info.get("raw_values_used", {})
    decisions = gating_info.get("gate_decisions", [])
    lines.append(f"  Gating raw:   {raw_vals}")
    if decisions:
        for d in decisions:
            lines.append(f"    → {d}")

    constraints = log.get("constraints_applied", [])
    if constraints:
        for c in constraints:
            lines.append(f"    → {c}")

    deltas = log.get("deltas", {})
    gate_d = deltas.get("gate_vs_ml", {})
    con_d = deltas.get("constraint_vs_gate", {})
    non_zero_gate = {k: v for k, v in gate_d.items() if v != 0}
    non_zero_con = {k: v for k, v in con_d.items() if v != 0}
    if non_zero_gate:
        lines.append(f"  Δ gate-ml:    {non_zero_gate}")
    if non_zero_con:
        lines.append(f"  Δ constr-gate:{non_zero_con}")

    return "\n".join(lines)

