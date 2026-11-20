"""
features_task.py – Task fulfillment feature extraction for IELTS Writing Scorer v2.

Detects the IELTS task type from the prompt, measures how well the essay
covers the required components, and scores argument depth based on
causality chains and supporting examples.

Dependencies:
    re
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Constants – Task-type keyword patterns
# ---------------------------------------------------------------------------
_TASK_PATTERNS: dict[str, list[str]] = {
    "discuss_both": ["discuss both", "both views", "both sides"],
    "opinion": ["to what extent", "do you agree", "your opinion"],
    "advantages": ["advantages", "disadvantages", "benefits", "drawbacks"],
    "problem": ["problems", "causes", "solutions", "measures"],
}

# ---------------------------------------------------------------------------
# Constants – Coverage markers
# ---------------------------------------------------------------------------
_CONTRAST_MARKERS: list[str] = [
    "however", "on the other hand", "while", "whereas",
    "conversely", "alternatively",
]

_OPINION_MARKERS: list[str] = [
    "i believe", "in my opinion", "i think", "i agree",
    "i disagree", "my view",
]

_ADVANTAGE_MARKERS: list[str] = [
    "advantage", "benefit", "positive", "strength", "merit", "pro",
]

_DISADVANTAGE_MARKERS: list[str] = [
    "disadvantage", "drawback", "negative", "weakness", "con",
]

_PROBLEM_MARKERS: list[str] = [
    "problem", "issue", "challenge", "cause", "reason",
]

_SOLUTION_MARKERS: list[str] = [
    "solution", "measure", "approach", "strategy", "remedy",
]

# ---------------------------------------------------------------------------
# Constants – Argument depth
# ---------------------------------------------------------------------------
_CAUSALITY_PHRASES: list[str] = [
    "because", "therefore", "as a result", "consequently",
    "this means", "which means", "thus", "hence", "due to", "owing to",
    "since", "so that", "in order to", "leads to", "results in",
    "this stems from", "this brings about", "as a consequence",
    "the reason is", "the primary cause", "this accounts for", "giving rise to", "thereby"
]

_EXAMPLE_PHRASES: list[str] = [
    "for example", "for instance", "such as", "to illustrate",
    "a case in point", "one clear example", "a prominent example",
    "as an illustration", "take for example", "take for instance",
    "studies show", "research suggests", "statistics indicate", "according to",
    "evidence suggests", "research shows"
]

# Pre-compiled regexes (longest first for correct matching)
_CAUSALITY_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in sorted(_CAUSALITY_PHRASES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_EXAMPLE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in sorted(_EXAMPLE_PHRASES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\b[a-z']+\b")

_STOPWORDS_TASK: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "i", "you", "it",
    "to", "of", "and", "in", "that", "this", "for", "on", "be", "with",
    "as", "have", "has", "had", "do", "does", "did", "not", "but", "or",
    "at", "by", "from", "they", "we", "he", "she", "some", "should", "what",
    "do", "think", "your", "extent", "agree", "disagree",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _lower_has_any(text: str, phrases: list[str]) -> bool:
    """Return True if *text* contains any of the *phrases* (case-insensitive)."""
    low = text.lower()
    return any(p in low for p in phrases)


def _count_phrases(text: str, compiled_re: re.Pattern[str]) -> int:
    """Count non-overlapping matches of *compiled_re* in *text*."""
    return len(compiled_re.findall(text))


def _sentence_count(text: str) -> int:
    """Rudimentary sentence count."""
    sents = _SENTENCE_SPLIT_RE.split(text.strip())
    return max(len([s for s in sents if s.strip()]), 1)


# ---------------------------------------------------------------------------
# Task type detection
# ---------------------------------------------------------------------------
def _detect_task_type(prompt: str) -> str:
    """Detect IELTS task type from the prompt text.

    Returns one of: 'discuss_both', 'opinion', 'advantages', 'problem',
    'two_part', or 'unknown'.
    """
    if not prompt:
        return "unknown"

    low = prompt.lower()

    # Check explicit patterns first (order matters – more specific first)
    for task_type, keywords in _TASK_PATTERNS.items():
        if any(kw in low for kw in keywords):
            return task_type

    # Two-part: 2 or more question marks
    if prompt.count("?") >= 2:
        return "two_part"

    return "unknown"


# ---------------------------------------------------------------------------
# Task coverage
# ---------------------------------------------------------------------------
def _task_coverage(essay: str, task_type: str) -> float:
    """Score [0, 1] indicating how well the essay covers the task requirements.

    For each task type, checks whether the essay contains the expected
    structural signals (contrast, opinion markers, etc.).
    """
    if task_type == "discuss_both":
        has_contrast = _lower_has_any(essay, _CONTRAST_MARKERS)
        contrast_count = sum(1 for m in _CONTRAST_MARKERS if m in essay.lower())
        coverage = min(contrast_count / 2.0, 1.0) if has_contrast else 0.0
        return round(coverage, 2)

    if task_type == "opinion":
        has_opinion = _lower_has_any(essay, _OPINION_MARKERS)
        return 1.0 if has_opinion else 0.2

    if task_type == "advantages":
        has_adv = _lower_has_any(essay, _ADVANTAGE_MARKERS)
        has_dis = _lower_has_any(essay, _DISADVANTAGE_MARKERS)
        if has_adv and has_dis:
            return 1.0
        if has_adv or has_dis:
            return 0.5
        return 0.1

    if task_type == "problem":
        has_prob = _lower_has_any(essay, _PROBLEM_MARKERS)
        has_sol = _lower_has_any(essay, _SOLUTION_MARKERS)
        if has_prob and has_sol:
            return 1.0
        if has_prob or has_sol:
            return 0.5
        return 0.1

    # two_part, unknown
    return 0.5


# ---------------------------------------------------------------------------
# Argument depth
# ---------------------------------------------------------------------------
def _argument_depth(essay: str) -> dict[str, Any]:
    """Compute argument-depth & structural density metrics from the essay (6J-A).

    Returns:
        causality_count, example_count, causal_relation_density, example_density,
        claim_support_ratio, task2_length_shortfall, explanation_chain_ratio,
        argument_depth_score
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(essay) if s.strip()]
    sent_count = max(len(sentences), 1)
    word_count = len(_WORD_RE.findall(essay.lower()))

    causality_count = _count_phrases(essay, _CAUSALITY_RE)
    example_count = _count_phrases(essay, _EXAMPLE_RE)

    causal_relation_density = round(causality_count / sent_count, 4)
    example_density = round(example_count / sent_count, 4)

    # Claim support ratio: sentences that contain a cause OR an example marker
    supported_sents = 0
    for s in sentences:
        if _CAUSALITY_RE.search(s) or _EXAMPLE_RE.search(s):
            supported_sents += 1
    claim_support_ratio = round(supported_sents / sent_count, 4)

    task2_length_shortfall = max(0.0, float(250 - word_count))

    explanation_chain_ratio = min(
        (causality_count + example_count) / sent_count, 1.0
    )

    # Structured argument depth score blending causal density, example density, and support ratio
    raw_score = (
        0.35 * claim_support_ratio +
        0.35 * min(causal_relation_density / 0.3, 1.0) +
        0.30 * min(example_density / 0.2, 1.0)
    )
    argument_depth_score = round(min(max(raw_score, 0.0), 1.0), 4)

    return {
        "causality_count": causality_count,
        "example_count": example_count,
        "causal_relation_density": causal_relation_density,
        "example_density": example_density,
        "claim_support_ratio": claim_support_ratio,
        "task2_length_shortfall": task2_length_shortfall,
        "explanation_chain_ratio": round(explanation_chain_ratio, 4),
        "argument_depth_score": argument_depth_score,
    }


def _prompt_keyword_overlap(essay: str, prompt: str) -> float:
    """Topic-relevance proxy: fraction of prompt content-words found in the essay.

    Strips stopwords from the prompt, then counts how many unique content-words
    appear in the essay.  This measures topic relevance, NOT Task Response quality.
    NOTE: intentionally word-overlap only to avoid leakage from label-derived info.
    """
    if not prompt or not prompt.strip():
        return 0.0
    prompt_words = {w for w in _WORD_RE.findall(prompt.lower()) if w not in _STOPWORDS_TASK and len(w) > 2}
    if not prompt_words:
        return 0.0
    essay_words = set(_WORD_RE.findall(essay.lower()))
    matched = prompt_words & essay_words
    return round(len(matched) / len(prompt_words), 4)


def _detect_tangential_drift(essay: str, prompt: str) -> float:
    """Detect whether the essay drifts into a tangential / off-topic subject.
    
    Checks if the body paragraphs address the core specific subject of the prompt
    rather than just generic context words (e.g. 'university', 'students', 'people').
    Returns 1.0 if tangential drift is detected, else 0.0.
    """
    if not prompt or not essay:
        return 0.0
    
    # Body paragraphs
    paras = [p.strip() for p in essay.split("\n\n") if len(p.strip()) > 30]
    body_paras = paras[1:-1] if len(paras) >= 3 else paras
    body_text = " ".join(body_paras).lower()
    body_words = set(_WORD_RE.findall(body_text))
    
    # Extract specific subject nouns from prompt, excluding generic context words
    prompt_words = set(_WORD_RE.findall(prompt.lower())) - _STOPWORDS_TASK
    generic_words = {
        "people", "think", "believe", "agree", "disagree", "opinion", "discuss",
        "views", "write", "least", "words", "suggested", "time", "university",
        "students", "will", "future", "world", "society", "others", "both",
        "such", "like", "only", "give", "many", "good", "well", "make"
    }
    prompt_specific = {w for w in prompt_words if len(w) > 3 and w not in generic_words}
    
    # Common academic & domain synonyms
    domain_synonyms = {
        "subject": {"discipline", "disciplines", "major", "majors", "field", "fields", "course", "courses", "curriculum", "curricula", "degree", "degrees", "humanities", "arts"},
        "subjects": {"discipline", "disciplines", "major", "majors", "field", "fields", "course", "courses", "curriculum", "curricula", "degree", "degrees", "humanities", "arts"},
        "science": {"stem", "physics", "chemistry", "biology", "mathematics", "engineering", "sciences"},
        "technology": {"digital", "computing", "software", "artificial", "computational", "technologies", "it"},
    }
    
    # Check direct specific prompt word presence in body
    overlap = prompt_specific & body_words
    
    # Check domain synonyms
    synonym_hits = set()
    for k, syn_set in domain_synonyms.items():
        if k in prompt_specific:
            synonym_hits |= (syn_set & body_words)
                
    total_matches = len(overlap | synonym_hits)
    
    # If specific prompt entities exist but body paragraphs have 0 matches,
    # the candidate has drifted off-topic into a tangential subject.
    if len(prompt_specific) > 0 and total_matches == 0:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_task_features(essay: str, prompt: str) -> dict[str, Any]:
    """Extract task-fulfilment features from an IELTS essay.

    Args:
        essay: Raw essay text.
        prompt: The IELTS writing prompt / task question.

    Returns:
        Dictionary with task_type, task_coverage_score,
        causality_count, example_count, causal_relation_density, example_density,
        claim_support_ratio, task2_length_shortfall, explanation_chain_ratio,
        argument_depth_score, prompt_keyword_overlap, is_tangential_drift.
    """
    if not essay or not essay.strip():
        return {
            "task_type": _detect_task_type(prompt),
            "task_coverage_score": 0.0,
            "causality_count": 0,
            "example_count": 0,
            "causal_relation_density": 0.0,
            "example_density": 0.0,
            "claim_support_ratio": 0.0,
            "task2_length_shortfall": 250.0,
            "explanation_chain_ratio": 0.0,
            "argument_depth_score": 0.0,
            "prompt_keyword_overlap": 0.0,
            "is_tangential_drift": 0.0,
        }

    task_type = _detect_task_type(prompt)
    coverage = _task_coverage(essay, task_type)
    depth = _argument_depth(essay)
    keyword_overlap = _prompt_keyword_overlap(essay, prompt)
    tangential_drift = _detect_tangential_drift(essay, prompt)

    return {
        "task_type": task_type,
        "task_coverage_score": coverage,
        "prompt_keyword_overlap": keyword_overlap,
        "is_tangential_drift": tangential_drift,
        **depth,
    }

