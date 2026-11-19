"""
train_model_f.py - Train Model F (Enhanced Features, L2, No Weight)
for Phase 6E comparison against Baseline Model A.
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm

from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import normalize_features
from scorer.ml_model import IELTSModel

def main():
    dataset_path = Path("data/train.csv")
    print(f"Loading {dataset_path}...")
    train_data = _load_dataset(dataset_path)

    features_list = []
    y_true_overall = []

    print("Extracting features (27 features)...")
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

    print(f"\nFeature count: {X_train.shape[1]}")
    print(f"Sample count:  {X_train.shape[0]}")
    print(f"Feature keys:  {FEATURE_KEYS}")

    print("\nTraining Model F (L2, No Weight)...")
    model = IELTSModel()
    model.train(
        X=X_train,
        y=y_dict,
        feature_names=FEATURE_KEYS,
        objective="regression",
        sample_weight=None,
    )
    model.save("data/model_exp_F.pkl")
    print("Saved to data/model_exp_F.pkl")

if __name__ == "__main__":
    main()
