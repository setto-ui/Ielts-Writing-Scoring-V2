"""
train_model_6jc.py – Train Model 6J-C (34 features: 31 6J-B baseline + 3 cohesion/structure features).
Computes feature_stats.json on train.csv and trains LightGBM model model_exp_6JC.pkl.
STRICT RULE: test.csv is LOCKED.
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import compute_stats_from_dataset, normalize_features
from scorer.ml_model import IELTSModel


def main():
    print("=" * 80)
    print("  TRAINING MODEL 6J-C (34 FEATURES: 31 6J-B BASELINE + 3 COHESION/STRUCTURE)")
    print("=" * 80)

    train_data = _load_dataset(Path("data/train.csv"))
    print(f"Loaded {len(train_data)} training samples.")
    print(f"Feature count registered in FEATURE_KEYS: {len(FEATURE_KEYS)}")

    # 1. Extract raw features for stats computation
    print("\nExtracting features for normalization stats...")
    all_raw_features = []
    for sample in tqdm(train_data, desc="Stats extraction"):
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        if not essay.strip() or not prompt.strip():
            continue
        raw_feat = extract_all_features(essay, prompt)
        all_raw_features.append(raw_feat)

    # 2. Compute and save feature stats for normalizer
    stats = compute_stats_from_dataset(all_raw_features)
    stats_path = Path("data/feature_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Updated {stats_path} with {len(stats)} feature stats.")

    # 3. Build normalized training feature matrix
    print("\nBuilding training feature matrix...")
    features_list = []
    y_list = []

    for sample, raw_feat in zip(train_data, all_raw_features):
        band = float(sample.get("band", 5.0))
        norm_feat = normalize_features(raw_feat)
        arr = features_to_array(norm_feat, FEATURE_KEYS)
        features_list.append(arr)
        y_list.append(band)

    X = np.vstack(features_list)
    y_dict = {"overall": np.array(y_list)}
    print(f"Training matrix shape: {X.shape}")

    # 4. Train IELTSModel (LightGBM)
    model = IELTSModel()
    model.train(X=X, y=y_dict, feature_names=FEATURE_KEYS, objective="regression", sample_weight=None)

    output_path = Path("data/model_exp_6JC.pkl")
    model.save(output_path)
    print(f"\nModel 6J-C saved to {output_path}")


if __name__ == "__main__":
    main()
