import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from scorer.ml_model import IELTSModel
from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import normalize_features
from scorer.scorer import round_to_ielts_band

def get_metrics(true_bands, pred_bands):
    errors = np.abs(np.array(pred_bands) - np.array(true_bands))
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    bias = np.mean(np.array(pred_bands) - np.array(true_bands))
    acc_05 = np.mean(errors <= 0.5) * 100
    return mae, rmse, bias, acc_05

def print_metrics_for_subset(title, true_bands, pred_bands):
    if len(true_bands) == 0:
        return
    mae, rmse, bias, acc_05 = get_metrics(true_bands, pred_bands)
    print(f"  {title:15s} | N: {len(true_bands):>3} | MAE: {mae:.3f} | Bias: {bias:>6.3f} | ±0.5: {acc_05:5.1f}%")

def evaluate_model(model_name, model_path, dataset):
    print(f"\n============================================================")
    print(f"  Model {model_name}")
    print(f"============================================================")
    
    if not model_path.exists():
        print(f"Model {model_path} not found.")
        return
        
    model = IELTSModel.load(model_path)
    
    true_bands = []
    pred_bands = []
    
    for sample in dataset:
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        true_band = sample.get("band", 5.0)

        if not essay.strip() or not prompt.strip():
            continue

        raw_features = extract_all_features(essay, prompt)
        norm_features = normalize_features(raw_features)
        X = features_to_array(norm_features, FEATURE_KEYS).reshape(1, -1)
        
        ml_preds = model.predict(X)
        raw_overall = float(ml_preds["overall"])
        final_band = round_to_ielts_band(raw_overall)
        
        true_bands.append(true_band)
        pred_bands.append(final_band)
        
    # Overall
    mae, rmse, bias, acc_05 = get_metrics(true_bands, pred_bands)
    print(f"  Overall MAE:      {mae:.3f}")
    print(f"  Overall RMSE:     {rmse:.3f}")
    print(f"  Overall Bias:     {bias:.3f}")
    print(f"  ±0.5 Accuracy:    {acc_05:.1f}%")
    
    print("\n  Group Metrics:")
    # Subsets
    low_idx = [i for i, b in enumerate(true_bands) if 4.0 <= b <= 5.5]
    mid_idx = [i for i, b in enumerate(true_bands) if 6.0 <= b <= 7.0]
    high_idx = [i for i, b in enumerate(true_bands) if 7.5 <= b <= 9.0]
    
    print_metrics_for_subset("Low (4.0-5.5)", [true_bands[i] for i in low_idx], [pred_bands[i] for i in low_idx])
    print_metrics_for_subset("Mid (6.0-7.0)", [true_bands[i] for i in mid_idx], [pred_bands[i] for i in mid_idx])
    print_metrics_for_subset("High (7.5-9.0)", [true_bands[i] for i in high_idx], [pred_bands[i] for i in high_idx])
    
    print("\n  Per-band Metrics:")
    print(f"  {'Band':>5} | {'N':>4} | {'MAE':>5} | {'Bias':>6} | {'±0.5 Acc':>8}")
    print("  " + "-" * 42)
    
    bands_dict = {}
    for t, p in zip(true_bands, pred_bands):
        if t not in bands_dict:
            bands_dict[t] = []
        bands_dict[t].append(p)
        
    for b in sorted(bands_dict.keys()):
        preds = bands_dict[b]
        n = len(preds)
        errs = [abs(p - b) for p in preds]
        b_mae = np.mean(errs)
        b_bias = np.mean([p - b for p in preds])
        b_acc = sum(1 for e in errs if e <= 0.5) / n * 100
        print(f"  {b:>5.1f} | {n:>4} | {b_mae:>5.3f} | {b_bias:>6.3f} | {b_acc:>7.1f}%")

def main():
    dataset_path = Path("data/val.csv")
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        sys.exit(1)

    print("Loading val.csv...")
    dataset = _load_dataset(dataset_path)

    experiments = [
        ("A (L2, No weight)", Path("data/model_exp_A.pkl")),
        ("B (MAE, No weight)", Path("data/model_exp_B.pkl")),
        ("C (Huber, No weight)", Path("data/model_exp_C.pkl")),
        ("D (L2, Tail-weight)", Path("data/model_exp_D.pkl")),
        ("E (Huber, Tail-weight)", Path("data/model_exp_E.pkl"))
    ]

    for name, path in experiments:
        evaluate_model(name, path, dataset)

if __name__ == "__main__":
    main()
