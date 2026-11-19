"""
two_stage_model.py — Tri-Specialist Architecture for IELTS Scoring.
Solves extreme band regression-to-the-mean using soft probability blending
between a Generalist Regressor and Low/High Specialist Regressors.
"""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import Any

_BAND_MIN = 4.0
_BAND_MAX = 9.0

class TriSpecialistModel:
    FEATURE_VERSION: str = "2.2"

    def __init__(self, alpha_low: float = 0.6, alpha_high: float = 0.6) -> None:
        self.feature_names: list[str] = []
        self.base_reg: lgb.LGBMRegressor | None = None
        self.clf_low: lgb.LGBMClassifier | None = None
        self.clf_high: lgb.LGBMClassifier | None = None
        self.reg_low: lgb.LGBMRegressor | None = None
        self.reg_high: lgb.LGBMRegressor | None = None
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        self.feature_names = list(feature_names)
        X_df = pd.DataFrame(X, columns=self.feature_names)

        # 1. Base Generalist Regressor
        print("  Training Base Generalist Regressor...")
        self.base_reg = lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_child_samples=3, num_leaves=15, objective="regression",
            verbosity=-1, random_state=42
        )
        self.base_reg.fit(X_df, y)

        # 2. Binary Classifiers with balanced weights
        print("  Training Low/High Binary Classifiers (class_weight='balanced')...")
        self.clf_low = lgb.LGBMClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.08,
            min_child_samples=5, class_weight="balanced", verbosity=-1, random_state=42
        )
        self.clf_low.fit(X_df, (y <= 5.5).astype(int))

        self.clf_high = lgb.LGBMClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.08,
            min_child_samples=5, class_weight="balanced", verbosity=-1, random_state=42
        )
        self.clf_high.fit(X_df, (y >= 7.0).astype(int))

        # 3. Specialist Regressors
        print("  Training Low Specialist Regressor (<= 5.5)...")
        idx_low = y <= 5.5
        self.reg_low = lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_child_samples=3, num_leaves=15, verbosity=-1, random_state=42
        )
        self.reg_low.fit(X_df.iloc[idx_low], y[idx_low])

        print("  Training High Specialist Regressor (>= 7.0)...")
        idx_high = y >= 7.0
        self.reg_high = lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_child_samples=3, num_leaves=15, verbosity=-1, random_state=42
        )
        self.reg_high.fit(X_df.iloc[idx_high], y[idx_high])

    def predict(self, X: np.ndarray) -> dict[str, float | np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)

        X_df = pd.DataFrame(X, columns=self.feature_names)

        p_base = self.base_reg.predict(X_df)
        p_low_spec = self.reg_low.predict(X_df)
        p_high_spec = self.reg_high.predict(X_df)

        prob_low = self.clf_low.predict_proba(X_df)[:, 1]
        prob_high = self.clf_high.predict_proba(X_df)[:, 1]

        w_l = prob_low * self.alpha_low
        w_h = prob_high * self.alpha_high
        w_b = np.maximum(0.0, 1.0 - w_l - w_h)
        total_w = w_b + w_l + w_h

        blended = (w_b * p_base + w_l * p_low_spec + w_h * p_high_spec) / total_w
        blended = np.clip(blended, _BAND_MIN, _BAND_MAX)

        res = float(np.squeeze(blended)) if single else blended
        # Return dict matching IELTSModel interface
        return {"overall": res}

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_reg": self.base_reg,
            "clf_low": self.clf_low,
            "clf_high": self.clf_high,
            "reg_low": self.reg_low,
            "reg_high": self.reg_high,
            "feature_names": self.feature_names,
            "alpha_low": self.alpha_low,
            "alpha_high": self.alpha_high,
            "feature_version": self.FEATURE_VERSION,
        }
        with open(out, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "TriSpecialistModel":
        p = Path(path)
        with open(p, "rb") as fh:
            payload = pickle.load(fh)
        inst = cls(alpha_low=payload.get("alpha_low", 0.6), alpha_high=payload.get("alpha_high", 0.6))
        inst.base_reg = payload["base_reg"]
        inst.clf_low = payload["clf_low"]
        inst.clf_high = payload["clf_high"]
        inst.reg_low = payload["reg_low"]
        inst.reg_high = payload["reg_high"]
        inst.feature_names = payload["feature_names"]
        return inst
