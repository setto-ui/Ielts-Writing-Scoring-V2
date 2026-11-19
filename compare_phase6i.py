"""
compare_phase6i.py - Phase 6I: Evaluation of Task 2-Only Model F vs Mixed Model F

Evaluates on data/val.csv (190 samples):
  - Task 2 Validation Subset (104 samples)
  - Task 1 Validation Subset (86 samples)
  - All Validation Samples (190 samples)

Models Compared:
  1. Model F (Mixed): Trained on 886 mixed Task 1 + Task 2 samples
  2. Model F (Task 2 Only): Trained on 504 Task 2 samples only

test.csv is STRICTLY LOCKED.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from scorer.ml_model import IELTSModel
from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import normalize_features
from scorer.scorer import round_to_ielts_band

TASK1_KEYWORDS = [
    "summarise the information", "selecting and reporting", "make comparisons",
    "the diagram", "the chart", "the table", "the graph", "the map",
    "the process", "the illustration", "the plan", "student accommodation",
    "bar chart", "line graph", "pie chart", "flow chart"
]

def is_task1(prompt: str) -> bool:
    p_lower = prompt.lower()
    return any(kw in p_lower for kw in TASK1_KEYWORDS)

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    bias = np.mean(y_pred - y_true)
    acc_05 = np.mean(np.abs(y_pred - y_true) <= 0.5) * 100.0
    return mae, rmse, bias, acc_05

def print_breakdown(title, y_true, y_pred):
    mae, rmse, bias, acc_05 = calculate_metrics(y_true, y_pred)
    print(f"--- {title} (N={len(y_true)}) ---")
    print(f"  MAE: {mae:.3f} | RMSE: {rmse:.3f} | Bias: {bias:+.3f} | ±0.5 Acc: {acc_05:.1f}%")

    low_idx  = [i for i, b in enumerate(y_true) if 4.0 <= b <= 5.5]
    mid_idx  = [i for i, b in enumerate(y_true) if 6.0 <= b <= 7.0]
    high_idx = [i for i, b in enumerate(y_true) if 7.5 <= b <= 9.0]

    def sub_line(label, idxs):
        if not idxs:
            print(f"  {label:<15}: N=0")
            return
        yt = [y_true[i] for i in idxs]
        yp = [y_pred[i] for i in idxs]
        m, r, b, a = calculate_metrics(yt, yp)
        print(f"  {label:<15}: N={len(idxs):>3} | MAE: {m:.3f} | Bias: {b:+.3f} | ±0.5 Acc: {a:.1f}%")

    sub_line("  Low (4.0-5.5)", low_idx)
    sub_line("  Mid (6.0-7.0)", mid_idx)
    sub_line("  High (7.5-9.0)", high_idx)
    print()

def main():
    val_path = Path("data/val.csv")
    print(f"Loading validation set from {val_path}...")
    val_data = _load_dataset(val_path)

    print("Loading models...")
    model_f_mixed = IELTSModel.load("data/model_exp_F.pkl")
    model_f_t2 = IELTSModel.load("data/model_exp_F_task2.pkl")

    y_true_all, y_pred_mixed_all, y_pred_t2_all = [], [], []
    task1_mask = []

    print("Extracting features for validation samples...")
    for sample in tqdm(val_data):
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        band = float(sample.get("band", 5.0))
        if not essay.strip() or not prompt.strip():
            continue

        raw_feat = extract_all_features(essay, prompt)
        norm_feat = normalize_features(raw_feat)
        X_arr = features_to_array(norm_feat, FEATURE_KEYS).reshape(1, -1)

        raw_mixed = float(model_f_mixed.models["overall"].predict(X_arr)[0])
        raw_t2 = float(model_f_t2.models["overall"].predict(X_arr)[0])

        pred_mixed = round_to_ielts_band(raw_mixed)
        pred_t2 = round_to_ielts_band(raw_t2)

        y_true_all.append(band)
        y_pred_mixed_all.append(pred_mixed)
        y_pred_t2_all.append(pred_t2)
        task1_mask.append(is_task1(prompt))

    y_true_all = np.array(y_true_all)
    y_pred_mixed_all = np.array(y_pred_mixed_all)
    y_pred_t2_all = np.array(y_pred_t2_all)
    task1_mask = np.array(task1_mask)
    task2_mask = ~task1_mask

    print("\n" + "=" * 75)
    print("  PHASE 6I EXPERIMENT RESULTS: TASK 2-ONLY vs MIXED MODEL F")
    print("=" * 75)

    print("\n[SECTION 1: TASK 2 VALIDATION SUBSET (N=104)] - CORE EVALUATION TARGET")
    print("=========================================================================")
    print_breakdown("Model F (Mixed Train 886) on Val Task 2", y_true_all[task2_mask], y_pred_mixed_all[task2_mask])
    print_breakdown("Model F (Task 2-Only Train 504) on Val Task 2", y_true_all[task2_mask], y_pred_t2_all[task2_mask])

    print("\n[SECTION 2: ALL VALIDATION SAMPLES (N=190)]")
    print("=========================================================================")
    print_breakdown("Model F (Mixed Train 886) on All Val", y_true_all, y_pred_mixed_all)
    print_breakdown("Model F (Task 2-Only Train 504) on All Val", y_true_all, y_pred_t2_all)

    print("\n[SECTION 3: TASK 1 VALIDATION SUBSET (N=86)]")
    print("=========================================================================")
    print_breakdown("Model F (Mixed Train 886) on Val Task 1", y_true_all[task1_mask], y_pred_mixed_all[task1_mask])
    print_breakdown("Model F (Task 2-Only Train 504) on Val Task 1", y_true_all[task1_mask], y_pred_t2_all[task1_mask])

    # Direct Comparison Table for Task 2 Subset
    yt_t2 = y_true_all[task2_mask]
    yp_m_t2 = y_pred_mixed_all[task2_mask]
    yp_t2_t2 = y_pred_t2_all[task2_mask]

    m_m, r_m, b_m, a_m = calculate_metrics(yt_t2, yp_m_t2)
    m_t, r_t, b_t, a_t = calculate_metrics(yt_t2, yp_t2_t2)

    print("=" * 75)
    print("  TASK 2 VALIDATION SUBSET DIRECT COMPARISON SUMMARY (N=104)")
    print("=" * 75)
    print(f"  {'Metric':<22} | {'Model F (Mixed)':>16} | {'Model F (Task 2)':>16} | {'Delta':>10}")
    print("  " + "-" * 70)
    print(f"  {'Overall MAE':<22} | {m_m:>16.3f} | {m_t:>16.3f} | {m_t - m_m:>+10.3f}")
    print(f"  {'Overall RMSE':<22} | {r_m:>16.3f} | {r_t:>16.3f} | {r_t - r_m:>+10.3f}")
    print(f"  {'Overall Bias':<22} | {b_m:>+16.3f} | {b_t:>+16.3f} | {b_t - b_m:>+10.3f}")
    print(f"  {'±0.5 Accuracy':<22} | {a_m:>15.1f}% | {a_t:>15.1f}% | {a_t - a_m:>+9.1f}%")
    print("  " + "-" * 70)

    # Per band bias on Task 2
    low_idx_t2  = [i for i, b in enumerate(yt_t2) if 4.0 <= b <= 5.5]
    mid_idx_t2  = [i for i, b in enumerate(yt_t2) if 6.0 <= b <= 7.0]
    high_idx_t2 = [i for i, b in enumerate(yt_t2) if 7.5 <= b <= 9.0]

    def get_bias_acc(yt, yp, idxs):
        if not idxs:
            return 0.0, 0.0, 0
        sub_yt = [yt[i] for i in idxs]
        sub_yp = [yp[i] for i in idxs]
        _, _, bias, acc = calculate_metrics(sub_yt, sub_yp)
        return bias, acc, len(idxs)

    b_low_m, a_low_m, n_low = get_bias_acc(yt_t2, yp_m_t2, low_idx_t2)
    b_low_t, a_low_t, _     = get_bias_acc(yt_t2, yp_t2_t2, low_idx_t2)

    b_mid_m, a_mid_m, n_mid = get_bias_acc(yt_t2, yp_m_t2, mid_idx_t2)
    b_mid_t, a_mid_t, _     = get_bias_acc(yt_t2, yp_t2_t2, mid_idx_t2)

    b_high_m, a_high_m, n_high = get_bias_acc(yt_t2, yp_m_t2, high_idx_t2)
    b_high_t, a_high_t, _      = get_bias_acc(yt_t2, yp_t2_t2, high_idx_t2)

    print(f"  {'Low Bias (N=' + str(n_low) + ')':<22} | {b_low_m:>+16.3f} | {b_low_t:>+16.3f} | {b_low_t - b_low_m:>+10.3f}")
    print(f"  {'Low ±0.5 Acc':<22} | {a_low_m:>15.1f}% | {a_low_t:>15.1f}% | {a_low_t - a_low_m:>+9.1f}%")
    print("  " + "-" * 70)
    print(f"  {'Mid Bias (N=' + str(n_mid) + ')':<22} | {b_mid_m:>+16.3f} | {b_mid_t:>+16.3f} | {b_mid_t - b_mid_m:>+10.3f}")
    print(f"  {'Mid ±0.5 Acc':<22} | {a_mid_m:>15.1f}% | {a_mid_t:>15.1f}% | {a_mid_t - a_mid_m:>+9.1f}%")
    print("  " + "-" * 70)
    print(f"  {'High Bias (N=' + str(n_high) + ')':<22} | {b_high_m:>+16.3f} | {b_high_t:>+16.3f} | {b_high_t - b_high_m:>+10.3f}")
    print(f"  {'High ±0.5 Acc':<22} | {a_high_m:>15.1f}% | {a_high_t:>15.1f}% | {a_high_t - a_high_m:>+9.1f}%")
    print("=" * 75)

if __name__ == "__main__":
    main()
