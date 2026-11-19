"""
diagnostic_engine.py
Analyzes normalized features to estimate sub-scores (Mode A), identify strengths/weaknesses,
and calculate tutoring priorities with targeted band-score improvements.
"""

from __future__ import annotations
import numpy as np

# Feature groups mapping to IELTS criteria
FEATURE_GROUPS = {
    "task_response": [
        "task_coverage_score",
        "argument_depth_score",
        "topic_similarity",
        "word_count",
    ],
    "coherence": [
        "semantic_coherence",
        "discourse_marker_variety",
        "discourse_marker_count",
        "connectors_per_sentence",
        "paragraph_count",
        "has_intro",
        "has_conclusion",
    ],
    "lexical": [
        "mattr",
        "rare_word_ratio",
        "awl_ratio",
        "max_word_repetition_ratio",
    ],
    "grammar": [
        "grammar_error_rate",
        "complex_clause_ratio",
        "sent_length_variance",
        "tense_consistent",
    ],
}

# Heuristic descriptions and target guidelines for each criterion
TUTORING_DETAILS = {
    "task_response": {
        "name": "Task Response",
        "description": "Focus on answering all parts of the question prompt and developing arguments with supporting examples.",
    },
    "coherence": {
        "name": "Coherence & Cohesion",
        "description": "Improve paragraph structure, sentence transitions, and use varied discourse markers to connect ideas.",
    },
    "lexical": {
        "name": "Lexical Resource",
        "description": "Expand vocabulary range, use precise academic terminology (AWL), and avoid repeating the same words.",
    },
    "grammar": {
        "name": "Grammatical Range & Accuracy",
        "description": "Minimize grammatical errors, ensure consistent verb tenses, and use complex clause structures.",
    },
}


def analyze(
    normalized_features: dict[str, float],
    raw_features: dict[str, float],
    overall_pred: float,
    is_estimated: bool,
    sub_scores: dict[str, float] | None = None,
) -> dict:
    """Analyze feature groups to determine strengths, weaknesses, and priorities.

    Parameters
    ----------
    normalized_features:
        Features normalized to [0, 1] (higher is better).
    raw_features:
        Raw un-normalized feature values.
    overall_pred:
        Calibrated overall band predicted by ML.
    is_estimated:
        True if sub-scores are estimated from features (Mode A), False if predicted (Mode B).
    sub_scores:
        Pre-computed sub-scores if already predicted (Mode B).

    Returns
    -------
    dict
        Structured analysis containing sub_scores, strengths, weaknesses, and priorities.
    """
    # 1. Estimate sub-scores if in Mode A
    if is_estimated or sub_scores is None:
        estimated_scores = _estimate_subscores_from_features(normalized_features, overall_pred)
    else:
        estimated_scores = {k: float(v) for k, v in sub_scores.items() if k != "overall"}

    # 2. Identify Strengths and Weaknesses
    strengths = _identify_strengths(normalized_features, raw_features)
    weaknesses = _identify_weaknesses(normalized_features, raw_features)

    # 3. Determine Tutoring Priorities
    priorities = _determine_priorities(estimated_scores, overall_pred)

    return {
        "sub_scores": estimated_scores,
        "is_estimated": is_estimated,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "priorities": priorities,
        "raw_features": raw_features,
    }


def _estimate_subscores_from_features(
    norm_feats: dict[str, float],
    overall_pred: float,
) -> dict[str, float]:
    """Estimate sub-scores relative to overall band based on feature group means."""
    group_means = {}
    for criterion, keys in FEATURE_GROUPS.items():
        vals = []
        for k in keys:
            if k in norm_feats:
                vals.append(norm_feats[k])
        group_means[criterion] = sum(vals) / len(vals) if vals else 0.5

    global_mean = sum(group_means.values()) / 4.0

    # Deviations from global average feature score
    deviations = {k: v - global_mean for k, v in group_means.items()}

    # Scale factor: a deviation of 0.2 maps to 1.0 band score difference
    scale = 5.0

    estimated = {}
    for criterion in FEATURE_GROUPS:
        dev = deviations[criterion]
        # Anchor around predicted overall score
        score = overall_pred + dev * scale
        # Clip to valid IELTS band range [4.0, 9.0]
        estimated[criterion] = float(np.clip(score, 4.0, 9.0))

    return estimated


def _identify_strengths(norm_feats: dict[str, float], raw_feats: dict[str, float]) -> list[str]:
    """Identify key writing strengths based on features exceeding thresholds."""
    strengths = []
    
    # Vocabulary diversity
    if norm_feats.get("mattr", 0) > 0.65:
        strengths.append("Excellent vocabulary diversity and lexical variety.")
    if norm_feats.get("rare_word_ratio", 0) > 0.65:
        strengths.append("Good usage of advanced or less-common vocabulary.")
        
    # Grammar
    if norm_feats.get("grammar_error_rate", 0) > 0.70:
        strengths.append("High grammatical accuracy with minimal errors.")
    if norm_feats.get("complex_clause_ratio", 0) > 0.65:
        strengths.append("Effective command of complex sentence structures.")
        
    # Coherence
    if norm_feats.get("semantic_coherence", 0) > 0.70:
        strengths.append("Strong logical coherence and smooth flow between ideas.")
    if norm_feats.get("discourse_marker_variety", 0) > 0.65:
        strengths.append("Varied and natural usage of cohesive discourse linkers.")
        
    # Task
    if raw_feats.get("word_count", 0) >= 280:
        strengths.append("Very appropriate length for IELTS Task 2 writing.")
    if norm_feats.get("task_coverage_score", 0) > 0.70:
        strengths.append("Good coverage and depth in addressing the task prompt.")

    # Fallback if no specific strengths detected
    if not strengths:
        strengths.append("Paragraph structuring meets basic organization requirements.")
        
    return strengths


def _identify_weaknesses(norm_feats: dict[str, float], raw_feats: dict[str, float]) -> list[str]:
    """Identify key writing weaknesses based on features failing to meet thresholds."""
    weaknesses = []

    # Word count
    wc = raw_feats.get("word_count", 0)
    if wc < 250:
        weaknesses.append(f"Essay length ({wc:.0f} words) is below the minimum required 250 words.")

    # Task Response
    if norm_feats.get("task_coverage_score", 0) < 0.40:
        weaknesses.append("Prompt requirements are only partially addressed or some viewpoints are missing.")
    if norm_feats.get("argument_depth_score", 0) < 0.35:
        weaknesses.append("Arguments lack sufficient development, explanations, or concrete examples.")
    if raw_feats.get("topic_similarity", 1.0) < 0.40:
        weaknesses.append("Weak topic focus: essay contains irrelevant points or deviates from the prompt.")

    # Coherence
    if raw_feats.get("paragraph_count", 4) < 4:
        weaknesses.append("Weak organization: essay should be structured into 4-5 clear paragraphs.")
    if norm_feats.get("semantic_coherence", 0) < 0.40:
        weaknesses.append("Sentences are poorly connected, making arguments difficult to follow.")
    if norm_feats.get("discourse_marker_variety", 0) < 0.35:
        weaknesses.append("Overuse of basic linking words or very low variety of cohesive devices.")

    # Lexical Resource
    if norm_feats.get("mattr", 0) < 0.40:
        weaknesses.append("Repetitive vocabulary: try to vary word choice and use synonyms.")
    if norm_feats.get("max_word_repetition_ratio", 1.0) < 0.35:
        weaknesses.append("Excessive repetition of specific words reduces lexical range.")

    # Grammar
    err_rate = raw_feats.get("grammar_error_rate", 0)
    if err_rate > 3.0:
        weaknesses.append(f"Frequent grammatical errors ({err_rate:.1f} per 100 words) interfere with clarity.")
    if norm_feats.get("complex_clause_ratio", 0) < 0.40:
        weaknesses.append("Simple sentence structures are overused; try integrating subordinate clauses.")

    return weaknesses


def _determine_priorities(scores: dict[str, float], overall_pred: float) -> list[dict]:
    """Calculate and order tutoring priorities based on sub-score deficits."""
    priorities = []
    
    for key, score in scores.items():
        # Deficit represents how far below overall band (or max possible) this criterion is
        deficit = max(0.0, overall_pred - score)
        
        # Calculate a realistic target band gain (step of 0.5 or 1.0)
        if deficit >= 1.25:
            gain = 1.5
        elif deficit >= 0.75:
            gain = 1.0
        elif deficit >= 0.25:
            gain = 0.5
        else:
            # Low deficit, target a small incremental improvement
            gain = 0.5 if score < 8.5 else 0.0

        details = TUTORING_DETAILS.get(key, {"name": key, "description": ""})
        
        priorities.append({
            "criterion_key": key,
            "criterion_name": details["name"],
            "current_score": score,
            "deficit": deficit,
            "estimated_gain": gain,
            "recommendation": details["description"],
        })

    # Sort priorities: highest deficit first
    priorities.sort(key=lambda x: x["deficit"], reverse=True)
    
    # Assign priority rank
    for rank, priority in enumerate(priorities):
        priority["priority_rank"] = rank + 1

    return priorities
