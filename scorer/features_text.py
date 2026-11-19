"""
features_text.py – Lexical and grammar feature extraction for IELTS Writing Scorer v2.

Extracts Moving-Average TTR, rare-word ratio, AWL ratio, word-repetition ratio,
grammar error counts/rates, clause complexity, sentence-length variance,
tense consistency, and top-5 grammar error details.

New features (Phase 6E):
    avg_sentence_length   – mean word count per sentence (proxy for syntactic complexity)
    sentence_length_std   – std dev of sentence lengths (proxy for sentence variety)
    grammar_error_density – grammar errors per sentence (more meaningful than per-word rate)
    long_word_ratio       – ratio of words with length >= 7 (crude proxy for lexical length)

Dependencies:
    spacy (en_core_web_sm), language_tool_python, wordfreq, re, collections
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any

import spacy
from language_tool_python import LanguageTool
from wordfreq import word_frequency

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_nlp: spacy.language.Language | None = None
_grammar_tool: LanguageTool | None = None


def _get_nlp() -> spacy.language.Language:
    """Return the shared spaCy pipeline, loading it on first call."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_grammar_tool():
    """Return the shared LanguageTool instance, or None if unavailable."""
    global _grammar_tool
    if _grammar_tool is None:
        try:
            _grammar_tool = LanguageTool("en-US")
        except Exception as exc:
            import warnings
            warnings.warn(
                f"LanguageTool unavailable ({exc}). "
                "Grammar errors will be estimated via spaCy only.",
                stacklevel=2,
            )
            _grammar_tool = False  # sentinel: tried and failed
    return _grammar_tool if _grammar_tool is not False else None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_STOPWORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "i", "you", "it",
    "to", "of", "and", "in", "that", "this", "for", "on", "be", "with",
    "as", "have", "has", "had", "do", "does", "did", "not", "but", "or",
    "at", "by", "from", "they", "we", "he", "she", "my", "your", "his",
    "her", "its", "our", "their",
}

_AWL_SET: set[str] = {
    "analysis", "approach", "area", "assess", "assume", "authority",
    "available", "benefit", "concept", "consistent", "constitute", "context",
    "contract", "create", "data", "define", "derive", "distribute",
    "economy", "environment", "establish", "estimate", "evidence", "export",
    "factor", "financial", "function", "identify", "income", "indicate",
    "individual", "interpret", "involve", "issue", "labour", "legal",
    "major", "method", "occur", "percent", "period", "policy", "principle",
    "process", "require", "research", "respond", "role", "section",
    "sector", "significant", "similar", "source", "structure", "theory",
    "variable", "volume", "promote", "crucial", "innovative", "implement",
    "impact", "generate", "enhance", "utilize", "obtain", "acquire",
    "achieve", "maintain", "sustain", "contribute", "demonstrate",
    "emphasize", "highlight", "illustrate", "reveal", "suggest", "support",
    "facilitate",
}

_TOKENIZE_RE = re.compile(r"\b[a-z']+\b")


# ---------------------------------------------------------------------------
# Tokeniser helper
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    """Lowercase regex tokenisation consistent across all feature modules."""
    return _TOKENIZE_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Lexical features
# ---------------------------------------------------------------------------
def mattr(words: list[str], window: int = 50) -> float:
    """Moving Average Type-Token Ratio.

    Slides a window of *window* tokens across *words*, computes TTR per
    window, and returns the mean.  Falls back to simple TTR when the token
    list is shorter than the window.

    Args:
        words: List of lowercased word tokens.
        window: Sliding-window size (default 50).

    Returns:
        MATTR value in the range [0, 1].
    """
    if not words:
        return 0.0
    if len(words) < window:
        return len(set(words)) / len(words)

    ttrs: list[float] = []
    for start in range(len(words) - window + 1):
        segment = words[start : start + window]
        ttrs.append(len(set(segment)) / window)
    return sum(ttrs) / len(ttrs) if ttrs else 0.0


def rare_word_ratio(words: list[str]) -> float:
    """Ratio of rare words (frequency < 1e-5 and length > 3) to total words."""
    if not words:
        return 0.0
    rare_count = sum(
        1
        for w in words
        if len(w) > 3 and word_frequency(w, "en") < 1e-5
    )
    return rare_count / len(words)


def awl_ratio(words: list[str]) -> float:
    """Proportion of tokens found in the Academic Word List."""
    if not words:
        return 0.0
    awl_count = sum(1 for w in words if w in _AWL_SET)
    return awl_count / len(words)


def max_word_repetition_ratio(words: list[str]) -> float:
    """Max repetition count of any non-stopword divided by total words.

    Returns 0.0 for empty input or when every token is a stopword.
    """
    if not words:
        return 0.0
    content_words = [w for w in words if w not in _STOPWORDS]
    if not content_words:
        return 0.0
    counter = Counter(content_words)
    return counter.most_common(1)[0][1] / len(words)


# ---------------------------------------------------------------------------
# Grammar features
# ---------------------------------------------------------------------------
_COMPLEX_DEPS: set[str] = {"advcl", "relcl", "ccomp", "xcomp", "acl"}

_PAST_TAGS: set[str] = {"VBD", "VBN"}
_PRESENT_TAGS: set[str] = {"VBP", "VBZ", "VBG"}
_FUTURE_LEMMAS: set[str] = {"will", "shall"}


def _grammar_features(essay: str, words: list[str]) -> dict[str, Any]:
    """Extract all grammar-related features from *essay*.

    Uses spaCy for syntactic analysis and LanguageTool for error detection.

    Returns a dict with keys:
        grammar_error_count, grammar_error_rate, complex_clause_ratio,
        sent_length_variance, tense_consistent, dominant_tense,
        grammar_errors_detail
    """
    nlp = _get_nlp()
    tool = _get_grammar_tool()
    word_count = len(words) if words else 1  # guard division

    # --- Grammar errors (LanguageTool or spaCy fallback) ---
    grammar_errors_detail: list[dict[str, Any]] = []

    if tool is not None:
        # LanguageTool available — use it
        matches = tool.check(essay) if essay.strip() else []
        grammar_error_count: int = len(matches)
        grammar_error_rate: float = grammar_error_count / word_count * 100

        for m in matches[:5]:
            grammar_errors_detail.append(
                {
                    "message": m.message,
                    "context": m.context,
                    "suggestions": m.replacements[:3] if m.replacements else [],
                }
            )
    else:
        # Fallback: estimate errors via spaCy (less accurate but functional)
        doc_for_err = nlp(essay) if essay.strip() else nlp("")
        error_count = 0
        for token in doc_for_err:
            # Heuristic: tokens that spaCy couldn't parse well
            if token.dep_ == "dep":  # unclassified dependency
                error_count += 1
            if token.tag_ == "" and token.is_alpha:  # no POS tag
                error_count += 1
        grammar_error_count = error_count
        grammar_error_rate = grammar_error_count / word_count * 100

    # --- spaCy parse ---
    doc = nlp(essay) if essay.strip() else nlp("")
    sentences = list(doc.sents)
    sentence_count = max(len(sentences), 1)

    # Complex clause ratio
    complex_dep_count = sum(
        1 for token in doc if token.dep_ in _COMPLEX_DEPS
    )
    complex_clause_ratio: float = complex_dep_count / sentence_count

    # Sentence-length variance
    sent_lengths = [len(sent) for sent in sentences]
    sent_length_variance: float = (
        statistics.variance(sent_lengths) if len(sent_lengths) >= 2 else 0.0
    )

    # Tense consistency
    past_count = 0
    present_count = 0
    future_count = 0
    for token in doc:
        if token.tag_ in _PAST_TAGS:
            past_count += 1
        elif token.tag_ in _PRESENT_TAGS:
            present_count += 1
        if token.lower_ in _FUTURE_LEMMAS:
            future_count += 1

    total_verb_tags = past_count + present_count + future_count
    if total_verb_tags == 0:
        dominant_tense = "unknown"
        tense_consistent = False
    else:
        tense_counts = {
            "past": past_count,
            "present": present_count,
            "future": future_count,
        }
        dominant_tense = max(tense_counts, key=tense_counts.get)  # type: ignore[arg-type]
        tense_consistent = (
            tense_counts[dominant_tense] / total_verb_tags >= 0.60
        )

    # --- New Phase 6E features ---
    # avg_sentence_length: mean words per sentence
    avg_sentence_length: float = (
        sum(sent_lengths) / sentence_count if sent_lengths else 0.0
    )

    # sentence_length_std: std dev of per-sentence word counts
    sentence_length_std: float = (
        statistics.stdev(sent_lengths) if len(sent_lengths) >= 2 else 0.0
    )

    # grammar_error_density: errors per sentence (distinct from per-word rate)
    grammar_error_density: float = grammar_error_count / sentence_count

    return {
        "grammar_error_count": grammar_error_count,
        "grammar_error_rate": round(grammar_error_rate, 2),
        "grammar_error_density": round(grammar_error_density, 4),
        "complex_clause_ratio": round(complex_clause_ratio, 2),
        "sent_length_variance": round(sent_length_variance, 2),
        "avg_sentence_length": round(avg_sentence_length, 4),
        "sentence_length_std": round(sentence_length_std, 4),
        "tense_consistent": tense_consistent,
        "dominant_tense": dominant_tense,
        "grammar_errors_detail": grammar_errors_detail,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_text_features(essay: str) -> dict[str, Any]:
    """Extract all lexical and grammar features from an IELTS essay.

    Args:
        essay: Raw essay text.

    Returns:
        Dictionary containing lexical metrics (mattr, rare_word_ratio,
        awl_ratio, max_word_repetition_ratio) and grammar metrics
        (error count/rate, clause complexity, sentence variance,
        tense consistency, top-5 error details).
    """
    if not essay or not essay.strip():
        return {
            "mattr": 0.0,
            "rare_word_ratio": 0.0,
            "awl_ratio": 0.0,
            "max_word_repetition_ratio": 0.0,
            "long_word_ratio": 0.0,
            "grammar_error_count": 0,
            "grammar_error_rate": 0.0,
            "grammar_error_density": 0.0,
            "complex_clause_ratio": 0.0,
            "sent_length_variance": 0.0,
            "avg_sentence_length": 0.0,
            "sentence_length_std": 0.0,
            "tense_consistent": False,
            "dominant_tense": "unknown",
            "grammar_errors_detail": [],
        }

    words = _tokenize(essay)

    # long_word_ratio: proxy for lexical length (NOT lexical sophistication).
    # Words with >= 7 characters tend to be less common, but this is a crude proxy.
    long_word_count = sum(1 for w in words if len(w) >= 7)
    long_word_ratio_val = long_word_count / len(words) if words else 0.0

    lexical: dict[str, Any] = {
        "mattr": round(mattr(words), 4),
        "rare_word_ratio": round(rare_word_ratio(words), 4),
        "awl_ratio": round(awl_ratio(words), 4),
        "max_word_repetition_ratio": round(max_word_repetition_ratio(words), 4),
        "long_word_ratio": round(long_word_ratio_val, 4),
    }

    grammar = _grammar_features(essay, words)
    return {**lexical, **grammar}
