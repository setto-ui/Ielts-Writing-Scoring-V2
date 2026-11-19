"""
ml_model.py – LightGBM multi-output regression for IELTS band prediction.

Four independent LGBMRegressor models are trained, one per IELTS criterion:
  * task_response
  * lexical
  * grammar
  * coherence

Hyper-parameters are deliberately conservative (shallow trees, few leaves) to
limit over-fitting on small training sets.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

# The four IELTS sub-score criterion names (fixed order).
CRITERIA: list[str] = ["overall", "task_response", "lexical", "grammar", "coherence"]

# Conservative LightGBM hyper-parameters for small-data training.
_LGB_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "min_child_samples": 3,
    "num_leaves": 15,
    "verbosity": -1,
}

# Band-score clipping range.
_BAND_MIN: float = 4.0
_BAND_MAX: float = 9.0


class IELTSModel:
    """Wrapper around LightGBM regressors (dynamic per criterion).

    Typical workflow::

        model = IELTSModel()
        model.train(X_train, y_train, feature_names)
        model.save("models/lgbm_model.pkl")

        # Later …
        model = IELTSModel.load("models/lgbm_model.pkl")
        predictions = model.predict(X_test)
    """
    FEATURE_VERSION: str = "2.1"

    def __init__(self) -> None:
        self.models: dict[str, lgb.LGBMRegressor] = {}
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: dict[str, list[float] | np.ndarray],
        feature_names: list[str],
        objective: str = "regression",
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Train one LGBMRegressor per target in y.

        Parameters
        ----------
        X:
            Array of shape ``(n_samples, n_features)`` with normalised features.
        y:
            Mapping ``{criterion_name: array_of_target_bands}`` where each
            array has length ``n_samples``.
        feature_names:
            Ordered feature names matching the columns of *X*.
        objective:
            LightGBM objective (e.g., 'regression', 'huber', 'mae').
        sample_weight:
            Optional array of shape ``(n_samples,)`` for sample weights.

        Raises
        ------
        ValueError
            If *y* is empty or shapes are inconsistent.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")

        if not y:
            raise ValueError("Target dictionary y cannot be empty.")

        self.feature_names = list(feature_names)
        self.models = {}

        params = _LGB_PARAMS.copy()
        params["objective"] = objective

        for criterion in y:
            targets = np.asarray(y[criterion], dtype=np.float64)
            if targets.shape[0] != X.shape[0]:
                raise ValueError(
                    f"Length mismatch for '{criterion}': "
                    f"X has {X.shape[0]} rows but y has {targets.shape[0]}."
                )

            reg = lgb.LGBMRegressor(**params)
            reg.fit(X, targets, sample_weight=sample_weight)
            self.models[criterion] = reg

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> dict[str, float]:
        """Predict IELTS scores for a single sample or batch.

        Parameters
        ----------
        X:
            Feature array.  Shape ``(n_features,)`` for a single essay or
            ``(n_samples, n_features)`` for a batch.  Feature order must match
            :attr:`feature_names` used during training.

        Returns
        -------
        dict[str, float]
            Predicted band scores clipped to [4.0, 9.0].  When *X* is a single
            sample the values are plain floats; for a batch each value is a
            :class:`numpy.ndarray`.

        Raises
        ------
        FileNotFoundError
            If no trained models are available.
        """
        if not self.models:
            raise FileNotFoundError(
                "No trained model available. "
                "Call IELTSModel.train() first or load a saved model."
            )

        import pandas as pd
        X = np.asarray(X, dtype=np.float64)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)

        # Wrap in DataFrame with feature names to suppress sklearn warning
        if self.feature_names:
            X_input = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_input = X

        results: dict[str, Any] = {}
        for criterion in self.models:
            preds = self.models[criterion].predict(X_input)
            preds = np.clip(preds, _BAND_MIN, _BAND_MAX)
            results[criterion] = float(preds[0]) if single else preds

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise trained models, feature names, and version to a pickle file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "models": self.models,
            "feature_names": self.feature_names,
            "feature_version": self.FEATURE_VERSION,
        }
        with open(out, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "IELTSModel":
        """Deserialise a previously saved model."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(
                f"Model file not found at '{p}'. Train a model first."
            )

        with open(p, "rb") as fh:
            payload: dict[str, Any] = pickle.load(fh)  # noqa: S301

        # Check feature version compatibility
        model_version = payload.get("feature_version", "unknown")
        if model_version != cls.FEATURE_VERSION:
            import warnings
            warnings.warn(
                f"Model feature version mismatch! Model has '{model_version}' "
                f"but current code uses '{cls.FEATURE_VERSION}'. This may cause failures.",
                UserWarning,
                stacklevel=2,
            )

        instance = cls()
        instance.models = payload["models"]
        instance.feature_names = payload["feature_names"]
        return instance
