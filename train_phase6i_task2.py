"""
train_phase6i_task2.py - Phase 6I: Task 2-Only Model F Training

Filters train.csv (886) for Task 2 essays only (~504 samples).
Trains Model F (27 features, L2 objective, conservative LightGBM hyperparameters).
Saves model to data/model_exp_F_task2.pkl.

Original train.csv, val.csv, and test.csv are preserved without modification.
test.csv remains STRICTLY LOCKED.
"""
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

from scorer.ml_model import IELTSModel
from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import normalize_features

# Task 1 detection keywords
TASK1_KEYWORDS = [
    "summarise the information", "selecting and reporting", "make comparisons",
    "the diagram", "the chart", "the table", "the graph", "the map",
    "the process", "the illustration", "the plan", "student accommodation",
    "bar chart", "line graph", "pie chart", "flow chart"
]

def is_task1(prompt: str) -> bool:
    p_lower = prompt.lower()
    return any(kw in p_lower for kw in TASK1_KEYWORDS)

def main():
    print("=" * 70)
    print("  PHASE 6I: TASK 2-ONLY MODEL F TRAINING")
    print("=" * 70)

    train_path = Path("data/train.csv")
    print(f"Loading {train_path}...")
    full_train = _load_dataset(train_path)

    # Filter for Task 2 only
    task2_train = []
    task1_count = 0
    for sample in full_train:
        prompt = sample.get("prompt", "")
        essay = sample.get("essay", "")
        if not essay.strip() or not prompt.strip():
            continue
        if is_task1(prompt):
            task1_count += 1
        else:
            task2_train.append(sample)

    print(f"Total Train Samples: {len(full_train)}")
    print(f"  Filtered out Task 1 : {task1_count} samples")
    print(f"  Retained Task 2 Only: {len(task2_train)} samples")

    # Extract 27 features
    X_list, y_list = [], []
    print("\nExtracting 27 features for Task 2 training set...")
    for sample in tqdm(task2_train):
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        band = float(sample.get("band", 5.0))

        raw_feat = extract_all_features(essay, prompt)
        norm_feat = normalize_features(raw_feat)
        arr = features_to_array(norm_feat, FEATURE_KEYS)
        
        X_list.append(arr)
        y_list.append(band)

    X_train = np.vstack(X_list)
    y_train = np.array(y_list)

    print(f"\nTask 2 Training Matrix: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Train IELTSModel (LightGBM L2) on Task 2 only
    model = IELTSModel()
    # Train overall criterion model
    import lightgbm as lgb
    lgb_params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "min_child_samples": 3,
        "num_leaves": 15,
        "objective": "regression",
        "verbosity": -1,
    }
    reg = lgb.LGBMRegressor(**lgb_params)
    reg.fit(X_train, y_train)
    model.models["overall"] = reg
    model.feature_names = FEATURE_KEYS

    output_path = Path("data/model_exp_F_task2.pkl")
    model.save(output_path)
    print(f"\nSaved Task 2-Only Model F to {output_path}")

    # Training MAE
    preds = reg.predict(X_train)
    train_mae = np.mean(np.abs(preds - y_train))
    print(f"  Task 2 Train MAE: {train_mae:.3f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
