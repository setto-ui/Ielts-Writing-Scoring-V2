"""
confidence.py
Heuristic confidence engine for scoring model predictions.
Assesses essay anomalies based on length, topic deviation, and feature outliers.
"""

from __future__ import annotations


def calculate_confidence(
    raw_features: dict[str, float],
    normalized_features: dict[str, float],
    overall_pred: float,
) -> float:
    """Calculate the prediction confidence score in [0.0, 1.0].

    Parameters
    ----------
    raw_features:
        Merged raw feature values.
    normalized_features:
        Features normalized to [0, 1].
    overall_pred:
        Calibrated overall band predicted by ML.

    Returns
    -------
    float
        Confidence score between 0.0 and 1.0.
    """
    confidence = 1.0

    # 1. Word count check (extreme lengths are out of typical training distribution)
    word_count = raw_features.get("word_count", 250)
    if word_count < 100:
        confidence -= 0.4  # Extremely short, prediction is very uncertain
    elif word_count < 180:
        confidence -= 0.15  # Too short
    elif word_count > 650:
        confidence -= 0.2  # Extremely long

    # 2. Topic similarity check (very low similarity implies completely off-topic or gibberish)
    topic_sim = raw_features.get("topic_similarity", 1.0)
    if topic_sim < 0.25:
        confidence -= 0.35
    elif topic_sim < 0.35:
        confidence -= 0.15

    # 3. Outlier check (if many normalized features are near 0 or 1, the essay is anomalous)
    outlier_count = 0
    for name, norm_val in normalized_features.items():
        if name.startswith("_"):
            continue
        # Values outside 0.02 to 0.98 are extreme (more than ~3 std dev)
        if norm_val < 0.02 or norm_val > 0.98:
            outlier_count += 1

    if outlier_count > 6:
        confidence -= 0.20
    elif outlier_count > 3:
        confidence -= 0.10

    # Clip to [0.0, 1.0]
    return max(0.0, min(1.0, float(confidence)))
