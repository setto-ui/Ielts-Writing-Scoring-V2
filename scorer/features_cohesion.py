"""
features_cohesion.py – Cohesion and Paragraph Structure features for IELTS Writing Scorer v2.

Phase 6J-C features:
1. connective_density: Density of discourse markers / connectives per 100 words in essay.
2. paragraph_structure_balance: 1.0 - (std(paragraph_wcs) / (mean(paragraph_wcs) + 1e-5)) for >=2 paragraphs.
3. lexical_overlap: Mean Jaccard similarity of non-stopword lemmas between adjacent paragraphs.

Dependencies:
    spacy (en_core_web_sm), numpy, re
"""

from __future__ import annotations

import re
from typing import Any
import numpy as np

from scorer.features_text import _get_nlp, _tokenize
from scorer.features_essay import _split_paragraphs, _DISCOURSE_RE


def extract_cohesion_features(essay: str) -> dict[str, float]:
    """
    Extract cohesion and structural balance features from an essay.

    Args:
        essay: Raw text of the essay.

    Returns:
        Dict with keys:
            - connective_density
            - paragraph_structure_balance
            - lexical_overlap
    """
    if not essay.strip():
        return {
            "connective_density": 0.0,
            "paragraph_structure_balance": 0.0,
            "lexical_overlap": 0.0,
        }

    words = _tokenize(essay)
    word_count = len(words)

    # 1. Connective Density (discourse markers per 100 words)
    if word_count > 0:
        matches = _DISCOURSE_RE.findall(essay)
        connective_density = (len(matches) / word_count) * 100.0
    else:
        connective_density = 0.0

    # Paragraph splitting
    paragraphs = _split_paragraphs(essay)
    num_paragraphs = len(paragraphs)

    # Edge cases for paragraph-level features (num_paragraphs < 2)
    if num_paragraphs < 2:
        return {
            "connective_density": round(connective_density, 4),
            "paragraph_structure_balance": 0.0,
            "lexical_overlap": 0.0,
        }

    # 2. Paragraph Structure Balance
    para_wcs = [len(_tokenize(p)) for p in paragraphs]
    mean_wc = float(np.mean(para_wcs))
    std_wc = float(np.std(para_wcs))

    if mean_wc > 0:
        para_balance = 1.0 - (std_wc / (mean_wc + 1e-5))
    else:
        para_balance = 0.0

    # 3. Lexical Overlap between adjacent paragraphs (Jaccard on non-stopword lemmas)
    nlp = _get_nlp()
    para_lemmas: list[set[str]] = []

    for p in paragraphs:
        doc = nlp(p)
        lemmas = {
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop
        }
        para_lemmas.append(lemmas)

    jaccards: list[float] = []
    for i in range(num_paragraphs - 1):
        p1 = para_lemmas[i]
        p2 = para_lemmas[i + 1]
        union = p1 | p2
        if union:
            jaccards.append(len(p1 & p2) / len(union))
        else:
            jaccards.append(0.0)

    lexical_overlap = float(np.mean(jaccards)) if jaccards else 0.0

    return {
        "connective_density": round(connective_density, 4),
        "paragraph_structure_balance": round(para_balance, 4),
        "lexical_overlap": round(lexical_overlap, 4),
    }
