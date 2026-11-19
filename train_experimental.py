import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import normalize_features
from scorer.ml_model import IELTSModel

def calculate_weights(y_true, base_weight=1.0, tail_weight=3.0):
    weights = np.ones(len(y_true)) * base_weight
    for i, band in enumerate(y_true):
        if band <= 5.5 or band >= 7.5:
            weights[i] = tail_weight
    return weights

def train_and_save_model(X, y_dict, obj, w, model_path):
    print(f"Training with objective='{obj}', weights={w is not None}...")
    model = IELTSModel()
    model.train(
        X=X,
        y=y_dict,
        feature_names=FEATURE_KEYS,
        objective=obj,
        sample_weight=w
    )
    model.save(model_path)
    print(f"Saved to {model_path}")

def main():
    dataset_path = Path("data/train.csv")
    print("Loading dataset...")
    train_data = _load_dataset(dataset_path)

    # 1. Feature extraction
    features_list = []
    y_true_overall = []
    
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
        y_true_overall.append(band)

    X_train = np.vstack(features_list)
    y_dict = {"overall": np.array(y_true_overall)}

    # 2. Prepare sample weights
    w_tail = calculate_weights(y_true_overall, base_weight=1.0, tail_weight=3.0)

    # 3. Train 5 models
    experiments = [
        ("A", "regression", None),           # L2, No weight
        ("B", "mae", None),                  # L1/MAE, No weight
        ("C", "huber", None),                # Huber, No weight
        ("D", "regression", w_tail),         # L2, Tail-weight
        ("E", "huber", w_tail)               # Huber, Tail-weight
    ]

    for name, obj, w in experiments:
        model_path = Path(f"data/model_exp_{name}.pkl")
        train_and_save_model(X_train, y_dict, obj, w, model_path)
        
    print("All experiments trained successfully.")

if __name__ == "__main__":
    main()
