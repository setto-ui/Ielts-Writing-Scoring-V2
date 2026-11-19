"""
features_antigaming.py – Gaming / manipulation detection for IELTS Writing Scorer v2.

Detects connector spam, AWL stuffing, paraphrase-only introductions, and
rare-word density abuse.  Produces a cumulative gaming penalty ∈ [0, 1].

Dependencies:
    re
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DISCOURSE_MARKERS: list[str] = [
    "however", "moreover", "furthermore", "in addition",
    "on the other hand", "for example", "such as", "therefore", "thus",
    "consequently", "nevertheless", "in conclusion", "to conclude",
    "to sum up", "firstly", "secondly", "finally", "additionally",
    "conversely", "similarly", "likewise", "in contrast", "meanwhile",
    "subsequently", "specifically", "notably", "whereas", "while",
    "although", "despite",
]

_DISCOURSE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in sorted(_DISCOURSE_MARKERS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

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

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\b[a-z']+\b")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    """Lowercase regex tokenisation."""
    return _WORD_RE.findall(text.lower())


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter on terminal punctuation."""
    sents = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sents if s.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines, falling back to single newlines."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(parts) > 1:
        return parts
    parts = [p.strip() for p in text.strip().split("\n") if p.strip()]
    return parts


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------
def _connector_spam(essay: str) -> dict[str, Any]:
    """Detect overuse of discourse markers per sentence."""
    sentences = _split_sentences(essay)
    sentence_count = max(len(sentences), 1)
    total_markers = len(_DISCOURSE_RE.findall(essay))
    connectors_per_sentence = total_markers / sentence_count
    connector_spam_flag = connectors_per_sentence > 2.5

    return {
        "connectors_per_sentence": round(connectors_per_sentence, 4),
        "connector_spam_flag": connector_spam_flag,
    }


def _awl_stuffing(essay: str) -> dict[str, Any]:
    """Detect excessive academic word list usage."""
    words = _tokenize(essay)
    word_count = max(len(words), 1)
    awl_count = sum(1 for w in words if w in _AWL_SET)
    awl_per_100 = awl_count / word_count * 100
    awl_stuffing_flag = awl_per_100 > 8.0

    return {
        "awl_per_100_tokens": round(awl_per_100, 2),
        "awl_stuffing_flag": awl_stuffing_flag,
    }


def _paraphrase_intro(essay: str, prompt: str) -> dict[str, Any]:
    """Detect if the introduction is merely a paraphrase of the prompt.

    Uses Jaccard similarity on word sets between the first paragraph and
    the prompt.  Flags when the first paragraph is very short AND the
    overlap exceeds 0.6.
    """
    if not prompt or not prompt.strip():
        return {"paraphrase_intro_flag": False}

    paragraphs = _split_paragraphs(essay)
    if not paragraphs:
        return {"paraphrase_intro_flag": False}

    first_para = paragraphs[0]
    first_words = set(_tokenize(first_para))
    prompt_words = set(_tokenize(prompt))

    jaccard_sim = _jaccard(first_words, prompt_words)
    # Flag only when intro is short-ish AND highly overlapping
    first_wc = len(_tokenize(first_para))
    paraphrase_intro_flag = first_wc < 60 and jaccard_sim > 0.6

    return {"paraphrase_intro_flag": paraphrase_intro_flag}


def _rare_stuffing(essay: str) -> dict[str, Any]:
    """Detect excessive use of rare words.

    Uses a lightweight heuristic: words longer than 3 characters that do
    not appear in a small common-words set (top ~3 000 English words are
    proxied by length + letter-frequency heuristics). For a more accurate
    check the caller can combine this with ``wordfreq`` in
    ``features_text.py``.  Here we keep it dependency-free by counting
    tokens with length > 10 or containing rare bigrams as a proxy.
    """
    # Lightweight proxy: any word with len > 10 is treated as "rare-ish"
    # This deliberately over-counts to flag suspicious density.
    words = _tokenize(essay)
    word_count = max(len(words), 1)
    rare_count = sum(1 for w in words if len(w) > 10)
    rare_per_100 = rare_count / word_count * 100
    rare_stuffing_flag = rare_per_100 > 10.0

    return {
        "rare_per_100_tokens": round(rare_per_100, 2),
        "rare_stuffing_flag": rare_stuffing_flag,
    }


# ---------------------------------------------------------------------------
# Gaming penalty
# ---------------------------------------------------------------------------
def _gaming_penalty(flags: dict[str, bool]) -> float:
    """Compute cumulative gaming penalty from individual flags.

    Each True flag adds 0.3 to the penalty, capped at 1.0.
    """
    penalty = sum(0.3 for v in flags.values() if v)
    return round(min(penalty, 1.0), 2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_antigaming_features(essay: str, prompt: str) -> dict[str, Any]:
    """Detect gaming / manipulation patterns in an IELTS essay.

    Args:
        essay: Raw essay text.
        prompt: The IELTS writing prompt / task question.

    Returns:
        Dictionary with connector-spam, AWL-stuffing, paraphrase-intro,
        rare-word-stuffing metrics and a cumulative gaming_penalty ∈ [0, 1].
    """
    if not essay or not essay.strip():
        return {
            "connectors_per_sentence": 0.0,
            "connector_spam_flag": False,
            "awl_per_100_tokens": 0.0,
            "awl_stuffing_flag": False,
            "paraphrase_intro_flag": False,
            "rare_per_100_tokens": 0.0,
            "rare_stuffing_flag": False,
            "gaming_penalty": 0.0,
        }

    connector = _connector_spam(essay)
    awl = _awl_stuffing(essay)
    paraphrase = _paraphrase_intro(essay, prompt)
    rare = _rare_stuffing(essay)

    all_flags = {
        "connector_spam": connector["connector_spam_flag"],
        "awl_stuffing": awl["awl_stuffing_flag"],
        "paraphrase_intro": paraphrase["paraphrase_intro_flag"],
        "rare_stuffing": rare["rare_stuffing_flag"],
    }
    penalty = _gaming_penalty(all_flags)

    return {
        **connector,
        **awl,
        **paraphrase,
        **rare,
        "gaming_penalty": penalty,
    }
