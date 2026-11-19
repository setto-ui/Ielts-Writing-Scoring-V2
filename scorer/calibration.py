"""
calibration.py – Isotonic-regression calibration for IELTS overall band.

After ML prediction, hard gating, and constraint enforcement, the mean of
the four sub-scores is passed through a trained
:class:`~sklearn.isotonic.IsotonicRegression` model to produce a calibrated
overall band.

If no calibration model has been trained yet the module falls back to simple
IELTS-band rounding (nearest 0.5 step).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression

# Default model path (relative to this file's parent).
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "calibration_model.pkl"

# Band-score clipping range.
_BAND_MIN: float = 4.0
_BAND_MAX: float = 9.0


# ---------------------------------------------------------------------------
# Band rounding
# ---------------------------------------------------------------------------

def round_to_ielts_band(score: float) -> float:
    """Round *score* to the nearest 0.5 within [4.0, 9.0].

    Examples
    --------
    >>> round_to_ielts_band(6.74)
    6.5
    >>> round_to_ielts_band(6.75)
    7.0
    """
    clamped = max(_BAND_MIN, min(_BAND_MAX, score))
    return round(clamped * 2) / 2


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_calibrator(
    raw_means: Sequence[float],
    true_bands: Sequence[float],
    save_path: str | None = None,
) -> IsotonicRegression:
    """Fit an isotonic-regression calibrator and optionally save it.

    Parameters
    ----------
    raw_means:
        Mean sub-scores (before calibration) for each training essay.
    true_bands:
        Ground-truth overall IELTS bands corresponding to *raw_means*.
    save_path:
        If given, serialise the fitted model to this path (parent directories
        are created automatically).

    Returns
    -------
    IsotonicRegression
        The fitted calibrator.

    Raises
    ------
    ValueError
        If inputs are empty or have mismatched lengths.
    """
    raw_arr = np.asarray(raw_means, dtype=np.float64)
    true_arr = np.asarray(true_bands, dtype=np.float64)

    if raw_arr.size == 0 or true_arr.size == 0:
        raise ValueError("raw_means and true_bands must be non-empty.")
    if raw_arr.shape != true_arr.shape:
        raise ValueError(
            f"Shape mismatch: raw_means {raw_arr.shape} vs true_bands {true_arr.shape}."
        )

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_arr, true_arr)

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(calibrator, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return calibrator


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def calibrate(
    raw_mean: float,
    model_path: str | None = None,
) -> float:
    """Calibrate a raw mean score into an IELTS band.

    Parameters
    ----------
    raw_mean:
        Arithmetic mean of the four constrained sub-scores.
    model_path:
        Path to a pickled :class:`IsotonicRegression` model.  Defaults to
        ``models/calibrator.pkl`` relative to this module.

    Returns
    -------
    float
        Calibrated IELTS band (rounded to nearest 0.5).

    Notes
    -----
    If the calibration model file does not exist the function falls back to
    simple IELTS-band rounding — no error is raised.
    """
    path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH

    if path.is_file():
        try:
            with open(path, "rb") as fh:
                calibrator: IsotonicRegression = pickle.load(fh)  # noqa: S301
            calibrated = float(calibrator.predict(np.array([raw_mean]))[0])
            return round_to_ielts_band(calibrated)
        except Exception:
            # Corrupted pickle or incompatible model → fall back gracefully.
            import warnings
            warnings.warn(
                f"Could not load calibrator from '{path}'. "
                "Falling back to simple rounding.",
                stacklevel=2,
            )

    # Fallback: plain rounding.
    return round_to_ielts_band(raw_mean)
