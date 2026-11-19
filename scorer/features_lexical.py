"""
features_lexical.py – Lexical Quality feature extraction for IELTS Writing Scorer v2.

Phase 6J-B features:
1. lexical_diversity: Raw TTR (unique lowercased lemmas / total alphabetic tokens)
2. lexical_density: Content words (NOUN, VERB, ADJ, ADV) / total alphabetic tokens
3. repetition_rate: Max content token frequency / total non-stop alphabetic tokens
4. academic_word_ratio: AWL tokens / total alphabetic tokens using existing AWL set

Dependencies:
    spacy (en_core_web_sm), collections.Counter
"""

from __future__ import annotations

from collections import Counter
from typing import Any
import spacy

from scorer.features_text import _get_nlp, _AWL_SET

_CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


def extract_lexical_features(essay: str) -> dict[str, float]:
    """
    Extract lexical quality features from an essay.

    Args:
        essay: Raw text of the essay.

    Returns:
        Dict with keys:
            - lexical_diversity
            - lexical_density
            - repetition_rate
            - academic_word_ratio
    """
    if not essay.strip():
        return {
            "lexical_diversity": 0.0,
            "lexical_density": 0.0,
            "repetition_rate": 0.0,
            "academic_word_ratio": 0.0,
        }

    nlp = _get_nlp()
    doc = nlp(essay)

    # Filter to alphabetic tokens
    alpha_tokens = [token for token in doc if token.is_alpha]
    total_alpha = len(alpha_tokens)

    if total_alpha == 0:
        return {
            "lexical_diversity": 0.0,
            "lexical_density": 0.0,
            "repetition_rate": 0.0,
            "academic_word_ratio": 0.0,
        }

    # 1. Lexical Diversity (Raw TTR on lowercased lemmas of alphabetic tokens)
    lemmas = [token.lemma_.lower() for token in alpha_tokens]
    lexical_diversity = len(set(lemmas)) / total_alpha

    # 2. Lexical Density (Content words: NOUN, VERB, ADJ, ADV / total alphabetic tokens)
    content_count = sum(1 for token in alpha_tokens if token.pos_ in _CONTENT_POS)
    lexical_density = content_count / total_alpha

    # 3. Repetition Rate (Max content token frequency / total non-stop alphabetic tokens)
    non_stop_alpha = [
        token.lemma_.lower()
        for token in alpha_tokens
        if not token.is_stop
    ]
    if non_stop_alpha:
        counts = Counter(non_stop_alpha)
        max_freq = max(counts.values())
        repetition_rate = max_freq / len(non_stop_alpha)
    else:
        repetition_rate = 0.0

    # 4. Academic Word Ratio (AWL tokens / total alphabetic tokens)
    awl_count = sum(
        1 for token in alpha_tokens
        if token.text.lower() in _AWL_SET or token.lemma_.lower() in _AWL_SET
    )
    academic_word_ratio = awl_count / total_alpha

    return {
        "lexical_diversity": round(lexical_diversity, 4),
        "lexical_density": round(lexical_density, 4),
        "repetition_rate": round(repetition_rate, 4),
        "academic_word_ratio": round(academic_word_ratio, 4),
    }
