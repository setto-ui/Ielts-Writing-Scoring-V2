"""
train_phase6f.py - Phase 6F: Classifier + Group-specific Regression (Hard Routing)

Architecture:
    27 Features
        ↓
    3-class LightGBM Classifier  (LOW: 4.0-5.5 / MID: 6.0-7.0 / HIGH: 7.5-9.0)
        ↓
    Hard Route → group-specific LightGBM Regressor
        ↓
    Final band (rounded)

Models saved:
    data/model_6f_clf.pkl        -- LightGBM Classifier
    data/model_6f_reg_low.pkl    -- Regressor trained on LOW band essays
    data/model_6f_reg_mid.pkl    -- Regressor trained on MID band essays
    data/model_6f_reg_high.pkl   -- Regressor trained on HIGH band essays
"""
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter

import lightgbm as lgb

from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import normalize_features

# Band-group mapping
def band_to_group(band: float) -> int:
    """0=LOW (4.0-5.5), 1=MID (6.0-7.0), 2=HIGH (7.5-9.0)"""
    if band <= 5.5:
        return 0
    elif band <= 7.0:
        return 1
    else:
        return 2

GROUP_NAMES = {0: "LOW (4.0-5.5)", 1: "MID (6.0-7.0)", 2: "HIGH (7.5-9.0)"}

# Shared LightGBM params (conservative, same as production model)
LGB_REG_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "min_child_samples": 3,
    "num_leaves": 15,
    "objective": "regression",
    "verbosity": -1,
}

LGB_CLF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "min_child_samples": 3,
    "num_leaves": 15,
    "objective": "multiclass",
    "num_class": 3,
    "verbosity": -1,
}


def main():
    dataset_path = Path("data/train.csv")
    print(f"Loading {dataset_path}...")
    train_data = _load_dataset(dataset_path)

    # 1. Feature extraction
    features_list = []
    y_bands = []

    print("Extracting features...")
    for sample in tqdm(train_data):
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        band = sample.get("band", 5.0)
        if not essay.strip() or not prompt.strip():
            continue
        raw_feat = extract_all_features(essay, prompt)
        norm_feat = normalize_features(raw_feat)
        arr = features_to_array(norm_feat, FEATURE_KEYS)
        features_list.append(arr)
        y_bands.append(band)

    X_all = np.vstack(features_list)
    y_all = np.array(y_bands)
    y_groups = np.array([band_to_group(b) for b in y_bands])

    print(f"\nDataset: {X_all.shape[0]} samples, {X_all.shape[1]} features")
    print("Group distribution:")
    dist = Counter(y_groups)
    for g in sorted(dist):
        print(f"  {GROUP_NAMES[g]}: {dist[g]} samples")

    # 2. Train Classifier (3-class)
    print("\nTraining 3-class Classifier...")
    clf = lgb.LGBMClassifier(**LGB_CLF_PARAMS)
    clf.fit(X_all, y_groups)
    with open("data/model_6f_clf.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("  Saved data/model_6f_clf.pkl")

    # Quick sanity: train accuracy
    clf_pred = clf.predict(X_all)
    clf_acc = np.mean(clf_pred == y_groups) * 100
    print(f"  Train accuracy (classifier): {clf_acc:.1f}%")

    # 3. Train group-specific regressors
    for group_id, group_name in GROUP_NAMES.items():
        idx = np.where(y_groups == group_id)[0]
        X_g = X_all[idx]
        y_g = y_all[idx]
        print(f"\nTraining regressor for {group_name} ({len(idx)} samples)...")
        reg = lgb.LGBMRegressor(**LGB_REG_PARAMS)
        reg.fit(X_g, y_g)
        path = f"data/model_6f_reg_{['low', 'mid', 'high'][group_id]}.pkl"
        with open(path, "wb") as f:
            pickle.dump(reg, f)
        # Train MAE
        preds = reg.predict(X_g)
        train_mae = np.mean(np.abs(preds - y_g))
        print(f"  Saved {path}  (train MAE: {train_mae:.3f})")

    print("\nPhase 6F training complete.")


if __name__ == "__main__":
    main()
