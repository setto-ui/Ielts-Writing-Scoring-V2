"""
rewrite.py
Grammar correction and structural improvement suggestions (rule-based, no LLM).
Uses LanguageTool for auto-correction.
"""

import language_tool_python

# ─── Lazy-loaded LanguageTool ─────────────────────────────

_tool = None


def _get_tool():
    global _tool
    if _tool is None:
        try:
            _tool = language_tool_python.LanguageTool("en-US")
        except Exception as e:
            return None
    return _tool


def rewrite_essay(essay: str, diagnostic_analysis: dict) -> dict:
    """
    Correct grammar and provide structural improvement suggestions.

    Args:
        essay: original essay text
        diagnostic_analysis: output dictionary from DiagnosticEngine

    Returns:
        {
            "corrected_essay": str,
            "corrections_made": list of correction dicts,
            "structural_suggestions": str
        }
    """
    if not essay or not essay.strip():
        return {
            "corrected_essay": "",
            "corrections_made": [],
            "structural_suggestions": "No essay provided.",
        }

    features = diagnostic_analysis.get("raw_features", {})
    corrected, corrections = _correct_grammar(essay)
    suggestions = _build_structural_suggestions(features)

    return {
        "corrected_essay": corrected,
        "corrections_made": corrections,
        "structural_suggestions": suggestions,
    }


# ─── Grammar Correction ──────────────────────────────────

def _correct_grammar(essay: str) -> tuple:
    """
    Apply LanguageTool corrections in reverse order to preserve offsets.

    Returns:
        (corrected_essay, list_of_corrections)
    """
    tool = _get_tool()
    if tool is None:
        return essay, [{"original": "", "corrected": "", "message": "Auto-correction disabled: LanguageTool unavailable", "offset": 0}]

    try:
        matches = tool.check(essay)
    except Exception as e:
        # Fallback if LanguageTool fails
        return essay, [{"original": "", "corrected": "", "message": f"Auto-correction disabled: {str(e)}", "offset": 0}]

    corrections = []
    corrected = essay

    # Apply in reverse to preserve character offsets
    for match in reversed(matches):
        if match.replacements:
            start = match.offset
            end = match.offset + match.errorLength
            original = corrected[start:end]
            replacement = match.replacements[0]

            try:
                corrected = corrected[:start] + replacement + corrected[end:]
                corrections.append({
                    "original": original,
                    "corrected": replacement,
                    "message": match.message,
                    "offset": match.offset,
                })
            except Exception:
                pass

    # Reverse so corrections are in document order
    corrections.reverse()

    return corrected, corrections


# ─── Structural Suggestions ──────────────────────────────

def _build_structural_suggestions(features: dict) -> str:
    """
    Generate a text block of structural improvement suggestions
    based on detected feature weaknesses.
    """
    suggestions = []

    # Introduction
    if not features.get("has_intro", True):
        suggestions.append(
            "📝 INTRODUCTION: Your essay lacks a clear introduction. "
            "Start with a sentence that paraphrases the question, then "
            "briefly outline your position or what the essay will discuss."
        )

    # Conclusion
    if not features.get("has_conclusion", True):
        suggestions.append(
            "📝 CONCLUSION: Add a conclusion paragraph that summarises your "
            "main arguments. Begin with a phrase like 'In conclusion', "
            "'To sum up', or 'Overall'."
        )

    # Paragraph structure
    para_count = features.get("paragraph_count", 4)
    if para_count < 3:
        suggestions.append(
            "📝 STRUCTURE: Your essay has only {} paragraph(s). "
            "Reorganise into at least 4 paragraphs: introduction, "
            "2 body paragraphs, and a conclusion.".format(para_count)
        )
    elif para_count == 3:
        suggestions.append(
            "📝 STRUCTURE: Consider splitting your body into 2 separate "
            "paragraphs — one for each main argument or viewpoint."
        )

    # Word count
    word_count = features.get("word_count", 250)
    if word_count < 250:
        deficit = 250 - word_count
        suggestions.append(
            f"📝 LENGTH: Your essay is {word_count} words — {deficit} words "
            f"below the recommended minimum of 250. Expand your arguments "
            f"with more examples and explanations."
        )

    # Argument depth
    if features.get("argument_depth_score", 1.0) < 0.3:
        suggestions.append(
            "📝 ARGUMENT DEVELOPMENT: Your arguments need more depth. "
            "For each main point, add:\n"
            "  1. A reason (because / therefore / as a result)\n"
            "  2. A specific example (for example / for instance / such as)\n"
            "  3. An explanation of why the example supports your point"
        )

    # Task coverage
    if features.get("task_coverage_score", 1.0) < 0.5:
        suggestions.append(
            "📝 TASK RESPONSE: You may not have addressed all parts of the "
            "question. Re-read the prompt carefully and ensure you:\n"
            "  - Discuss ALL viewpoints mentioned\n"
            "  - Give your own opinion (if asked)\n"
            "  - Address causes AND solutions (if asked)"
        )

    # Discourse markers
    if features.get("discourse_marker_variety", 1.0) < 0.15:
        suggestions.append(
            "📝 COHESION: Use more linking words to connect your ideas. "
            "Try: however, furthermore, consequently, in contrast, "
            "additionally, nevertheless, likewise."
        )

    if not suggestions:
        return "Your essay structure is solid. Focus on refining vocabulary and grammar for a higher band."

    return "\n\n".join(suggestions)
