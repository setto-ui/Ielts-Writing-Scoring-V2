"""
normalizer.py – Normalize raw IELTS features to [0, 1] via z-score + sigmoid.

Features where *lower* raw values indicate *better* writing are inverted
so that the normalized output always means "higher = better".

Stats (mean / std per feature) are loaded from ``data/feature_stats.json``
when available; otherwise a conservative DEFAULT_STATS fallback is used.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Default statistics (initial estimates – replaced by train.py after fitting)
# ---------------------------------------------------------------------------

DEFAULT_STATS: dict[str, dict[str, float]] = {
    "mattr":                    {"mean": 0.52,  "std": 0.08},
    "rare_word_ratio":          {"mean": 0.022, "std": 0.012},
    "awl_ratio":                {"mean": 0.015, "std": 0.008},
    "grammar_error_rate":       {"mean": 2.5,   "std": 1.5},
    "complex_clause_ratio":     {"mean": 1.2,   "std": 0.6},
    "sent_length_variance":     {"mean": 30.0,  "std": 20.0},
    "semantic_coherence":       {"mean": 0.60,  "std": 0.10},
    "topic_similarity":         {"mean": 0.55,  "std": 0.12},
    "discourse_marker_variety": {"mean": 0.25,  "std": 0.12},
    "paragraph_count":          {"mean": 4.0,   "std": 1.2},
    "word_count":               {"mean": 270,   "std": 60},
    "task_coverage_score":      {"mean": 0.6,   "std": 0.2},
    "argument_depth_score":     {"mean": 0.4,   "std": 0.2},
    "connectors_per_sentence":  {"mean": 0.8,   "std": 0.4},
    "max_word_repetition_ratio":{"mean": 0.04,  "std": 0.02},
}

# Features where a *lower* raw value means *better* writing.
# After sigmoid normalisation these are flipped: norm = 1 - sigmoid(z).
_INVERTED_FEATURES: set[str] = {
    "grammar_error_rate",
    "connectors_per_sentence",
    "max_word_repetition_ratio",
    "gaming_penalty",
}

# Resolve the stats JSON relative to this file's parent directory.
_STATS_PATH = Path(__file__).resolve().parent.parent / "data" / "feature_stats.json"


# ---------------------------------------------------------------------------
# Stats I/O
# ---------------------------------------------------------------------------

def _load_stats() -> dict[str, dict[str, float]]:
    """Load per-feature mean/std from JSON, falling back to defaults."""
    if _STATS_PATH.is_file():
        try:
            with open(_STATS_PATH, "r", encoding="utf-8") as fh:
                loaded: dict[str, Any] = json.load(fh)
            # Validate structure: every value must have "mean" and "std".
            for key, val in loaded.items():
                if not (isinstance(val, dict) and "mean" in val and "std" in val):
                    raise ValueError(
                        f"Invalid stats entry for '{key}': expected dict with 'mean' and 'std'."
                    )
            return loaded
        except (json.JSONDecodeError, ValueError) as exc:
            # Malformed file → fall back silently to defaults.
            import warnings
            warnings.warn(
                f"Could not parse {_STATS_PATH}: {exc}. Using DEFAULT_STATS.",
                stacklevel=2,
            )
    return DEFAULT_STATS.copy()


# Module-level cache so the file is read only once per process.
_CACHED_STATS: dict[str, dict[str, float]] | None = None


def get_stats() -> dict[str, dict[str, float]]:
    """Return the active feature statistics (cached after first call)."""
    global _CACHED_STATS  # noqa: PLW0603
    if _CACHED_STATS is None:
        _CACHED_STATS = _load_stats()
    return _CACHED_STATS


def reload_stats() -> dict[str, dict[str, float]]:
    """Force-reload statistics from disk (useful after re-training)."""
    global _CACHED_STATS  # noqa: PLW0603
    _CACHED_STATS = None
    return get_stats()


# ---------------------------------------------------------------------------
# Core normalisation
# ---------------------------------------------------------------------------

def zscore_sigmoid(value: float, mean: float, std: float) -> float:
    """Map *value* to (0, 1) via z-score → logistic sigmoid.

    If *std* is near-zero the distribution is degenerate and the function
    returns a neutral 0.5 to avoid division-by-zero.
    """
    if std < 1e-9:
        return 0.5
    z = (value - mean) / std
    return 1.0 / (1.0 + math.exp(-z))


def normalize_features(features: dict[str, float]) -> dict[str, float]:
    """Normalise every feature in *features* to [0, 1].

    * Standard features: ``norm = sigmoid(z)``
    * Inverted features (lower raw = better): ``norm = 1 - sigmoid(z)``

    Unknown features (not in the stats table) are passed through unchanged
    so downstream consumers can decide how to handle them.

    Parameters
    ----------
    features:
        Mapping of raw feature names to their numeric values.

    Returns
    -------
    dict[str, float]
        Same keys, all values in [0, 1].
    """
    if not features:
        return {}

    stats = get_stats()
    normalised: dict[str, float] = {}

    for name, value in features.items():
        if name.startswith("_"):
            # Skip metadata/debug keys
            continue
        if name in stats:
            s = stats[name]
            norm = zscore_sigmoid(float(value), s["mean"], s["std"])
            if name in _INVERTED_FEATURES:
                norm = 1.0 - norm
            normalised[name] = norm
        else:
            # Feature has no stats entry – pass through raw value.
            normalised[name] = float(value)

    return normalised


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def compute_stats_from_dataset(
    feature_dicts: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Compute per-feature mean and std from a list of raw feature dicts.

    Parameters
    ----------
    feature_dicts:
        Each element is a ``{feature_name: value}`` mapping (one per essay).

    Returns
    -------
    dict[str, dict[str, float]]
        ``{feature_name: {"mean": float, "std": float}}``
    """
    if not feature_dicts:
        raise ValueError("feature_dicts must be a non-empty list.")

    # Collect values per feature key.
    buckets: dict[str, list[float]] = {}
    for fd in feature_dicts:
        for key, val in fd.items():
            buckets.setdefault(key, []).append(float(val))

    stats: dict[str, dict[str, float]] = {}
    for key, values in buckets.items():
        arr = np.array(values, dtype=np.float64)
        stats[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),  # population std (full dataset)
        }
    return stats
