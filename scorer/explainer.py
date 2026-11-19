def explain(
    diagnostic_analysis: dict,
    features: dict,
    applied_constraints: list,
    gaming_flags: dict,
) -> dict:
    """
    Build a full tutoring feedback report.

    Args:
        diagnostic_analysis: Output dict from DiagnosticEngine
        features: merged raw feature dict
        applied_constraints: list of constraint rule names that fired
        gaming_flags: dict of gaming detection results

    Returns:
        dict with score_breakdown, anti_gaming_flags, constraints_applied,
        improvements, strengths, priorities
    """
    sub_scores = diagnostic_analysis["sub_scores"]
    is_estimated = diagnostic_analysis["is_estimated"]
    
    breakdown = _build_breakdown(sub_scores, features, is_estimated)
    
    # We use strengths and weaknesses directly from DiagnosticEngine
    strengths = diagnostic_analysis["strengths"]
    improvements = diagnostic_analysis["weaknesses"]
    priorities = diagnostic_analysis["priorities"]

    return {
        "score_breakdown": breakdown,
        "anti_gaming_flags": gaming_flags,
        "constraints_applied": applied_constraints,
        "improvements": improvements,
        "strengths": strengths,
        "priorities": priorities,
    }


# ─── Score Breakdown ──────────────────────────────────────

def _build_breakdown(scores: dict, feat: dict, is_estimated: bool) -> dict:
    subscore_type = "Estimated from feature analysis (heuristic diagnostic only)" if is_estimated else "Predicted by ML model"
    
    # Check if overall is in scores to display it as ML predicted
    breakdown = {}
    if "overall" in scores:
        breakdown["overall"] = {
            "score": scores.get("overall"),
            "type": "Predicted by ML model",
            "drivers": ["Calibrated LightGBM overall predictor"],
            "issues": [],
        }

    breakdown.update({
        "task_response": {
            "score": scores.get("task_response"),
            "type": subscore_type,
            "drivers": [
                f"Topic similarity: {feat.get('topic_similarity', 0):.2f}",
                f"Task coverage: {feat.get('task_coverage_score', 0):.2f}",
                f"Argument depth: {feat.get('argument_depth_score', 0):.2f}",
                f"Word count: {feat.get('word_count', 0)}",
            ],
            "issues": _task_issues(feat),
        },
        "lexical_resource": {
            "score": scores.get("lexical"),
            "type": subscore_type,
            "drivers": [
                f"Vocabulary diversity (MATTR): {feat.get('mattr', 0):.3f}",
                f"Rare word ratio: {feat.get('rare_word_ratio', 0):.3f}",
                f"Academic vocab ratio: {feat.get('awl_ratio', 0):.3f}",
                f"Max repetition ratio: {feat.get('max_word_repetition_ratio', 0):.3f}",
            ],
            "issues": _lexical_issues(feat),
        },
        "grammar_range_accuracy": {
            "score": scores.get("grammar"),
            "type": subscore_type,
            "drivers": [
                f"Error rate (per 100 words): {feat.get('grammar_error_rate', 0):.1f}",
                f"Complex clause ratio: {feat.get('complex_clause_ratio', 0):.2f}",
                f"Sentence length variance: {feat.get('sent_length_variance', 0):.1f}",
                f"Tense consistent: {feat.get('tense_consistent', True)}",
            ],
            "issues": _grammar_issues(feat),
        },
        "coherence_cohesion": {
            "score": scores.get("coherence"),
            "type": subscore_type,
            "drivers": [
                f"Semantic coherence: {feat.get('semantic_coherence', 0):.2f}",
                f"Discourse marker variety: {feat.get('discourse_marker_variety', 0):.2f}",
                f"Paragraph count: {feat.get('paragraph_count', 0)}",
                f"Has introduction: {feat.get('has_intro', False)}",
                f"Has conclusion: {feat.get('has_conclusion', False)}",
            ],
            "issues": _coherence_issues(feat),
        },
    })
    return breakdown


def _task_issues(f: dict) -> list:
    issues = []
    if f.get("topic_similarity", 1.0) < 0.45:
        issues.append("Essay may be off-topic or weakly related to the prompt")
    if f.get("task_coverage_score", 1.0) < 0.5:
        issues.append("Not all parts of the question are addressed")
    if f.get("argument_depth_score", 1.0) < 0.3:
        issues.append("Arguments lack supporting examples and explanations")
    if f.get("word_count", 250) < 250:
        issues.append(f"Below minimum word count ({f.get('word_count', 0)}/250)")
    return issues


def _lexical_issues(f: dict) -> list:
    issues = []
    if f.get("mattr", 1.0) < 0.45:
        issues.append("Vocabulary range is limited — too much word repetition")
    if f.get("rare_word_ratio", 1.0) < 0.01:
        issues.append("Overreliance on common/basic vocabulary")
    if f.get("max_word_repetition_ratio", 0) > 0.08:
        issues.append("Excessive repetition of specific words")
    return issues


def _grammar_issues(f: dict) -> list:
    issues = []
    rate = f.get("grammar_error_rate", 0)
    if rate > 3.0:
        issues.append("High number of grammar errors throughout the essay")
    elif rate > 1.5:
        issues.append("Noticeable grammar errors that reduce clarity")
    if f.get("complex_clause_ratio", 1.0) < 0.5:
        issues.append("Limited use of complex sentence structures")
    if not f.get("tense_consistent", True):
        issues.append("Inconsistent tense usage across the essay")
    return issues


def _coherence_issues(f: dict) -> list:
    issues = []
    if f.get("paragraph_count", 4) < 4:
        issues.append("Too few paragraphs — essay needs clearer organisation")
    if f.get("discourse_marker_variety", 1.0) < 0.15:
        issues.append("Very few cohesive devices used")
    if f.get("semantic_coherence", 1.0) < 0.5:
        issues.append("Ideas are not well connected between sentences")
    if not f.get("has_intro", True):
        issues.append("No clear introduction paragraph detected")
    if not f.get("has_conclusion", True):
        issues.append("No clear conclusion paragraph detected")
    return issues


# ─── Improvements ─────────────────────────────────────────

def _detect_improvements(scores: dict, f: dict) -> list:
    tips = []

    if f.get("grammar_error_rate", 0) > 2.0:
        tips.append(
            "Focus on grammar accuracy: reduce errors in subject-verb agreement, "
            "article usage, and verb tense forms."
        )
    if f.get("mattr", 1.0) < 0.45:
        tips.append(
            "Expand vocabulary range: use synonyms and paraphrasing to avoid "
            "repeating the same words."
        )
    if f.get("discourse_marker_variety", 1.0) < 0.2:
        tips.append(
            "Use more varied linking words and cohesive devices "
            "(e.g., however, furthermore, consequently, in contrast)."
        )
    if f.get("paragraph_count", 4) < 4:
        tips.append(
            "Organise your essay into 4–5 clear paragraphs: "
            "introduction, 2–3 body paragraphs, and conclusion."
        )
    if f.get("task_coverage_score", 1.0) < 0.5:
        tips.append(
            "Make sure to address ALL parts of the question. "
            "Re-read the prompt and check each requirement."
        )
    if f.get("argument_depth_score", 1.0) < 0.3:
        tips.append(
            "Develop your arguments with specific examples, reasons, "
            "and explanations (use 'for example', 'because', 'as a result')."
        )
    if f.get("word_count", 250) < 250:
        tips.append(
            f"Write at least 250 words for Task 2 "
            f"(current: {f.get('word_count', 0)} words)."
        )
    if not f.get("has_conclusion", True):
        tips.append(
            "Add a clear conclusion paragraph that summarises your main points "
            "(start with 'In conclusion', 'To sum up', or 'Overall')."
        )
    if not f.get("has_intro", True):
        tips.append(
            "Add an introduction that paraphrases the question and "
            "states your position or essay outline."
        )
    if f.get("complex_clause_ratio", 1.0) < 0.5:
        tips.append(
            "Use more complex sentences with subordinate clauses "
            "(e.g., 'Although…', 'While…', 'If…then…')."
        )

    return tips


# ─── Strengths ────────────────────────────────────────────

def _detect_strengths(scores: dict, f: dict) -> list:
    strong = []

    if f.get("mattr", 0) > 0.55:
        strong.append("Good vocabulary diversity")
    if f.get("grammar_error_rate", 99) < 1.0:
        strong.append("Strong grammar accuracy")
    if f.get("semantic_coherence", 0) > 0.7:
        strong.append("Well-connected ideas throughout the essay")
    if f.get("discourse_marker_variety", 0) > 0.3:
        strong.append("Effective use of cohesive devices")
    if f.get("has_intro", False) and f.get("has_conclusion", False):
        strong.append("Well-structured essay with clear introduction and conclusion")
    if f.get("word_count", 0) >= 280:
        strong.append("Appropriate essay length")
    if f.get("task_coverage_score", 0) >= 0.8:
        strong.append("All parts of the question are fully addressed")
    if f.get("argument_depth_score", 0) >= 0.6:
        strong.append("Arguments are well-developed with examples and reasoning")
    if f.get("rare_word_ratio", 0) > 0.03:
        strong.append("Good use of less common vocabulary")
    if f.get("complex_clause_ratio", 0) >= 1.5:
        strong.append("Excellent range of complex sentence structures")

    return strong
