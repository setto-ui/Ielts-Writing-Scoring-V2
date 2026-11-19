"""
train_model_6ja.py - Train Model 6J-A (31 Features: 27 Base + 4 Argument/Task Response)
Trains LightGBM with L2 loss on train.csv (886 samples) and saves to data/model_exp_6JA.pkl.
STRICT RULE: test.csv is LOCKED.
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm

from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import normalize_features, compute_stats_from_dataset
from scorer.ml_model import IELTSModel
import json

def main():
    dataset_path = Path("data/train.csv")
    print(f"Loading {dataset_path}...")
    train_data = _load_dataset(dataset_path)

    raw_features_list = []
    features_list = []
    y_true_overall = []

    print(f"Extracting features ({len(FEATURE_KEYS)} features)...")
    for sample in tqdm(train_data):
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        band = sample.get("band", 5.0)
        if not essay.strip() or not prompt.strip():
            continue
        raw_feat = extract_all_features(essay, prompt)
        raw_features_list.append(raw_feat)
        norm_feat = normalize_features(raw_feat)
        arr = features_to_array(norm_feat, FEATURE_KEYS)
        features_list.append(arr)
        y_true_overall.append(band)

    # Compute and save feature stats for normalizer
    stats = compute_stats_from_dataset(raw_features_list)
    stats_path = Path("data/feature_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Updated {stats_path} with {len(stats)} feature stats.")

    X_train = np.vstack(features_list)
    y_dict = {"overall": np.array(y_true_overall)}

    print(f"\nFeature count: {X_train.shape[1]}")
    print(f"Sample count:  {X_train.shape[0]}")
    print(f"Feature keys:  {FEATURE_KEYS}")

    print("\nTraining Model 6J-A (31 features, L2 regression)...")
    model = IELTSModel()
    model.train(
        X=X_train,
        y=y_dict,
        feature_names=FEATURE_KEYS,
        objective="regression",
        sample_weight=None,
    )
    output_model_path = "data/model_exp_6JA.pkl"
    model.save(output_model_path)
    print(f"Successfully saved Model 6J-A to {output_model_path}")

if __name__ == "__main__":
    main()
