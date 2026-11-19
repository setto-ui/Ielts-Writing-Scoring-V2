"""
features_essay.py – Essay structure and coherence features for IELTS Writing Scorer v2.

Extracts paragraph/sentence structure, intro/conclusion detection,
semantic coherence (consecutive-sentence similarity), topic similarity
to the prompt, and discourse-marker statistics.

Dependencies:
    sentence_transformers, numpy, re
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Lazy-loaded singleton
# ---------------------------------------------------------------------------
_st_model: SentenceTransformer | None = None


def _get_st_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer model, loading it on first call."""
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CONCLUSION_STARTERS: tuple[str, ...] = (
    "in conclusion",
    "to conclude",
    "to sum up",
    "overall",
    "in summary",
)

_COHESIVE_DEVICES: tuple[str, ...] = (
    "however", "moreover", "furthermore", "in addition", "therefore",
    "thus", "consequently", "nevertheless", "firstly", "secondly",
    "finally", "additionally", "similarly", "likewise", "meanwhile",
)

_DISCOURSE_MARKERS: list[str] = [
    "however", "moreover", "furthermore", "in addition",
    "on the other hand", "for example", "such as", "therefore", "thus",
    "consequently", "nevertheless", "in conclusion", "to conclude",
    "to sum up", "firstly", "secondly", "finally", "additionally",
    "conversely", "similarly", "likewise", "in contrast", "meanwhile",
    "subsequently", "specifically", "notably", "whereas", "while",
    "although", "despite",
]

# Pre-compile a single pattern that matches any discourse marker (longest first
# to avoid partial matches, e.g. "in addition" before "in").
_DISCOURSE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in sorted(_DISCOURSE_MARKERS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\b[a-z']+\b")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _split_paragraphs(essay: str) -> list[str]:
    """Split essay into paragraphs on blank lines OR single newlines.

    The training dataset uses single ``\\n`` for paragraph breaks (no
    ``\\n\\n``), so we first try double-newline splitting; if that yields
    only 1 block we fall back to single-newline splitting.
    """
    # Prefer double-newline split (real user essays)
    parts = [p.strip() for p in re.split(r"\n\s*\n", essay.strip()) if p.strip()]
    if len(parts) > 1:
        return parts
    # Fallback: single-newline split (training dataset format)
    parts = [p.strip() for p in essay.strip().split("\n") if p.strip()]
    return parts


def _split_sentences(text: str) -> list[str]:
    """Rudimentary sentence splitter by terminal punctuation."""
    sents = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sents if s.strip()]


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text.lower()))


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors, safe against zero norms."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Structure features
# ---------------------------------------------------------------------------
def _structure_features(essay: str) -> dict[str, Any]:
    """Paragraph count, intro/conclusion detection, word & sentence counts."""
    paragraphs = _split_paragraphs(essay)
    paragraph_count = len(paragraphs)
    sentences = _split_sentences(essay)
    sentence_count = max(len(sentences), 1)
    word_count = _word_count(essay)

    # Intro heuristic: first paragraph short AND not starting with cohesive device
    has_intro = False
    if paragraphs:
        first_wc = _word_count(paragraphs[0])
        first_lower = paragraphs[0].lower().lstrip()
        starts_with_cohesive = any(
            first_lower.startswith(cd) for cd in _COHESIVE_DEVICES
        )
        has_intro = first_wc < 80 and not starts_with_cohesive

    # Conclusion heuristic: last paragraph starts with conclusion phrase
    has_conclusion = False
    if paragraphs:
        last_lower = paragraphs[-1].lower().lstrip()
        has_conclusion = any(
            last_lower.startswith(cs) for cs in _CONCLUSION_STARTERS
        )

    return {
        "paragraph_count": paragraph_count,
        "has_intro": has_intro,
        "has_conclusion": has_conclusion,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_paragraph_length": round(word_count / paragraph_count, 4) if paragraph_count else 0.0,
    }


# ---------------------------------------------------------------------------
# Coherence features
# ---------------------------------------------------------------------------
def _coherence_features(essay: str, prompt: str) -> dict[str, Any]:
    """Semantic coherence and topic similarity using sentence-transformers."""
    model = _get_st_model()
    sentences = _split_sentences(essay)

    # --- Semantic coherence: mean cosine sim between consecutive sentences ---
    if len(sentences) < 2:
        semantic_coherence = 1.0  # single sentence is trivially coherent
    else:
        embeddings = model.encode(sentences, convert_to_numpy=True)
        sims: list[float] = []
        for i in range(len(embeddings) - 1):
            raw_sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
            # Normalise to [0, 1]
            sims.append((raw_sim + 1.0) / 2.0)
        semantic_coherence = float(np.mean(sims)) if sims else 1.0

    # --- Topic similarity: prompt ↔ full essay ---
    if not prompt or not prompt.strip():
        topic_similarity = 0.0
    else:
        pair_embs = model.encode([prompt, essay], convert_to_numpy=True)
        topic_similarity = float(_cosine_similarity(pair_embs[0], pair_embs[1]))
        topic_similarity = (topic_similarity + 1.0) / 2.0  # normalise

    return {
        "semantic_coherence": round(semantic_coherence, 4),
        "topic_similarity": round(topic_similarity, 4),
    }


# ---------------------------------------------------------------------------
# Discourse-marker features
# ---------------------------------------------------------------------------
def _discourse_features(essay: str) -> dict[str, Any]:
    """Count and variety of discourse markers in the essay."""
    matches = _DISCOURSE_RE.findall(essay)
    marker_count = len(matches)
    unique_used = len({m.lower() for m in matches})
    total_available = len(_DISCOURSE_MARKERS)

    return {
        "discourse_marker_count": marker_count,
        "discourse_marker_variety": round(unique_used / total_available, 4)
        if total_available
        else 0.0,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_essay_features(essay: str, prompt: str) -> dict[str, Any]:
    """Extract structure, coherence, and discourse features from an essay.

    Args:
        essay: Raw essay text.
        prompt: The IELTS writing prompt / task question.

    Returns:
        Dictionary of essay-level features including paragraph_count,
        has_intro, has_conclusion, word_count, sentence_count,
        semantic_coherence, topic_similarity, discourse_marker_count,
        and discourse_marker_variety.
    """
    if not essay or not essay.strip():
        return {
            "paragraph_count": 0,
            "has_intro": False,
            "has_conclusion": False,
            "word_count": 0,
            "sentence_count": 0,
            "avg_paragraph_length": 0.0,
            "semantic_coherence": 0.0,
            "topic_similarity": 0.0,
            "discourse_marker_count": 0,
            "discourse_marker_variety": 0.0,
        }

    structure = _structure_features(essay)
    coherence = _coherence_features(essay, prompt)
    discourse = _discourse_features(essay)

    return {**structure, **coherence, **discourse}
