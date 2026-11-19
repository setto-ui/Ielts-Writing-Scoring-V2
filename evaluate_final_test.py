"""
evaluate_final_test.py – ONE-TIME Final Test Evaluation on data/test.csv (191 samples).

Model under evaluation: Model 6J-B (31 features)
Norm stats: data/feature_stats.json (computed on train.csv)

Reports:
1. Overall Test Metrics (MAE, RMSE, Bias, ±0.5 Acc)
2. Task 2 Only Test Metrics (191 test samples breakdown)
3. Test Metrics by Band Bucket (Low, Mid, High-Mid, High)
4. Half-Band Confusion Matrix (True vs Pred)
5. Comparison Summary Across Splits (Train vs Val vs Test)
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix

from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import normalize_features
from scorer.ml_model import IELTSModel
from scorer.scorer import round_to_ielts_band


def compute_metrics(true_bands, pred_bands):
    tb = np.array(true_bands, dtype=np.float64)
    pb = np.array(pred_bands, dtype=np.float64)
    errors = np.abs(pb - tb)
    return {
        "mae": np.mean(errors),
        "rmse": np.sqrt(np.mean(errors ** 2)),
        "bias": np.mean(pb - tb),
        "acc05": np.mean(errors <= 0.5) * 100.0,
    }


def main():
    print("=" * 90)
    print("  [UNLOCKED] ONE-TIME FINAL TEST EVALUATION")
    print("  Model: Model 6J-B (31 Features) | Dataset: data/test.csv (191 Samples)")
    print("=" * 90)

    test_path = Path("data/test.csv")
    if not test_path.exists():
        print(f"Error: {test_path} not found.")
        return

    test_data = _load_dataset(test_path)
    print(f"Loaded {len(test_data)} locked test samples.\n")

    model_path = Path("data/model_exp_6JB.pkl")
    model = IELTSModel.load(str(model_path))

    true_bands = []
    pred_bands = []
    raw_preds = []
    task_types = []

    print("Running 31-feature pipeline on test set...")
    for sample in test_data:
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        band = float(sample.get("band", 5.0))
        if not essay.strip() or not prompt.strip():
            continue

        raw_feat = extract_all_features(essay, prompt)
        norm_feat = normalize_features(raw_feat)
        arr = features_to_array(norm_feat, FEATURE_KEYS).reshape(1, -1)
        r_pred = float(model.predict(arr)["overall"].flatten()[0])
        p_band = round_to_ielts_band(r_pred)

        wc = raw_feat.get("word_count", 0)

        true_bands.append(band)
        pred_bands.append(p_band)
        raw_preds.append(r_pred)
        task_types.append("Task 2" if wc >= 200 else "Task 1")

    # --- 1. OVERALL TEST METRICS ---
    m_test = compute_metrics(true_bands, pred_bands)
    print("\n1. OVERALL FINAL TEST METRICS (N=191, Rounded Bands):")
    print("-" * 70)
    print(f"  Test MAE:               {m_test['mae']:.3f}")
    print(f"  Test RMSE:              {m_test['rmse']:.3f}")
    print(f"  Test ±0.5 Accuracy:     {m_test['acc05']:.1f}%")
    print(f"  Test Overall Bias:      {m_test['bias']:+.3f}")
    print("-" * 70)

    # --- 2. TASK 2 ONLY TEST METRICS ---
    t2_idx = [i for i, t in enumerate(task_types) if t == "Task 2"]
    t2_true = [true_bands[i] for i in t2_idx]
    t2_pred = [pred_bands[i] for i in t2_idx]
    m_t2 = compute_metrics(t2_true, t2_pred) if t2_idx else {}

    print(f"\n2. TASK 2 ONLY TEST METRICS (N={len(t2_true)}, Rounded Bands):")
    print("-" * 70)
    print(f"  Task 2 Test MAE:        {m_t2.get('mae', 0):.3f}")
    print(f"  Task 2 Test RMSE:       {m_t2.get('rmse', 0):.3f}")
    print(f"  Task 2 Test ±0.5 Acc:   {m_t2.get('acc05', 0):.1f}%")
    print(f"  Task 2 Test Bias:       {m_t2.get('bias', 0):+.3f}")
    print("-" * 70)

    # --- 3. BAND BUCKET BREAKDOWN ---
    print("\n3. FINAL TEST METRICS BY BAND BUCKET:")
    print("-" * 75)
    print(f"{'Band Bucket':<20} | {'N':>4} | {'MAE':>6} | {'RMSE':>6} | {'Bias':>7} | {'±0.5 Acc':>8}")
    print("-" * 75)

    buckets = [
        ("Low (4.0-5.5)", 4.0, 5.5),
        ("Mid (6.0-6.5)", 6.0, 6.5),
        ("High-Mid (7.0-7.5)", 7.0, 7.5),
        ("High (8.0-9.0)", 8.0, 9.0),
    ]

    for name, low, high in buckets:
        idx = [i for i, b in enumerate(true_bands) if low <= b <= high]
        if idx:
            sub_t = [true_bands[i] for i in idx]
            sub_p = [pred_bands[i] for i in idx]
            sub_m = compute_metrics(sub_t, sub_p)
            print(f"{name:<20} | {len(idx):>4} | {sub_m['mae']:>6.3f} | {sub_m['rmse']:>6.3f} | {sub_m['bias']:>+7.3f} | {sub_m['acc05']:>7.1f}%")

    print("-" * 75)

    # --- 4. HALF-BAND CONFUSION MATRIX ---
    all_possible_bands = ["4.0", "4.5", "5.0", "5.5", "6.0", "6.5", "7.0", "7.5", "8.0", "8.5", "9.0"]
    true_str = [f"{b:.1f}" for b in true_bands]
    pred_str = [f"{b:.1f}" for b in pred_bands]
    cm = confusion_matrix(true_str, pred_str, labels=all_possible_bands)
    cm_df = pd.DataFrame(cm, index=[f"True {b}" for b in all_possible_bands],
                             columns=[f"Pred {b}" for b in all_possible_bands])

    print("\n4. FINAL TEST CONFUSION MATRIX (True vs Predicted Bands):")
    print("-" * 90)
    print(cm_df.to_string())
    print("-" * 90)

    # --- 5. SPLIT COMPARISON SUMMARY (Train vs Val vs Test) ---
    val_data = _load_dataset(Path("data/val.csv"))
    val_t, val_p = [], []
    for sample in val_data:
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        band = float(sample.get("band", 5.0))
        if not essay.strip() or not prompt.strip():
            continue
        raw_feat = extract_all_features(essay, prompt)
        norm_feat = normalize_features(raw_feat)
        arr = features_to_array(norm_feat, FEATURE_KEYS).reshape(1, -1)
        r_pred = float(model.predict(arr)["overall"].flatten()[0])
        val_t.append(band)
        val_p.append(round_to_ielts_band(r_pred))

    m_val = compute_metrics(val_t, val_p)

    high_val_idx = [i for i, b in enumerate(val_t) if b >= 7.5]
    m_val_high = compute_metrics([val_t[i] for i in high_val_idx], [val_p[i] for i in high_val_idx])

    high_test_idx = [i for i, b in enumerate(true_bands) if b >= 7.5]
    m_test_high = compute_metrics([true_bands[i] for i in high_test_idx], [pred_bands[i] for i in high_test_idx])

    print("\n5. FINAL BENCHMARK SUMMARY (Model 6J-B Across Splits):")
    print("=" * 80)
    print(f"  {'Metric':<24} | {'Validation (N=190)':>18} | {'Final Test (N=191)':>18}")
    print("  " + "-" * 64)
    print(f"  {'Overall MAE':<24} | {m_val['mae']:>18.3f} | {m_test['mae']:>18.3f}")
    print(f"  {'Overall RMSE':<24} | {m_val['rmse']:>18.3f} | {m_test['rmse']:>18.3f}")
    print(f"  {'Overall ±0.5 Acc':<24} | {m_val['acc05']:>17.1f}% | {m_test['acc05']:>17.1f}%")
    print(f"  {'Overall Bias':<24} | {m_val['bias']:>+18.3f} | {m_test['bias']:>+18.3f}")
    print(f"  {'High-Band (>=7.5) MAE':<24} | {m_val_high['mae']:>18.3f} | {m_test_high['mae']:>18.3f}")
    print(f"  {'High-Band (>=7.5) ±0.5':<24} | {m_val_high['acc05']:>17.1f}% | {m_test_high['acc05']:>17.1f}%")
    print(f"  {'High-Band (>=7.5) Bias':<24} | {m_val_high['bias']:>+18.3f} | {m_test_high['bias']:>+18.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
