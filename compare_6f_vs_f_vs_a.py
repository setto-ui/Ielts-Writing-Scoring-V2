"""
compare_6f_vs_f_vs_a.py - Evaluation of Phase 6F (Hard Routing & Soft Routing) vs Model F vs Model A

Evaluates on data/val.csv (190 clean samples, LOCKED test set unused).

Pipelines evaluated:
  1. Model A: Baseline LightGBM (21 features)
  2. Model F: Single LightGBM Regressor (27 features)
  3. Model F1 (Phase 6F - Hard Routing): 3-class Classifier -> Group Regressor
  4. Model F2 (Phase 6F - Soft Routing): P(Group) weighted sum of Group Regressors
"""
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import normalize_features
from scorer.ml_model import IELTSModel

FEATURE_KEYS_A = sorted([
    "argument_depth_score", "awl_ratio", "complex_clause_ratio",
    "connectors_per_sentence", "discourse_marker_count",
    "discourse_marker_variety", "gaming_penalty", "grammar_error_rate",
    "has_conclusion", "has_intro", "mattr", "max_word_repetition_ratio",
    "paragraph_count", "rare_word_ratio", "semantic_coherence",
    "sent_length_variance", "sentence_count", "task_coverage_score",
    "tense_consistent", "topic_similarity", "word_count",
])

# Band grouping helper
def band_to_group(band: float) -> int:
    if band <= 5.5:
        return 0
    elif band <= 7.0:
        return 1
    else:
        return 2

def get_band_group_label(band: float) -> str:
    if band <= 5.5:
        return "Low"
    elif band <= 7.0:
        return "Mid"
    else:
        return "High"

def round_to_ielts_band(val: float) -> float:
    # Round to nearest 0.5 step in range [4.0, 9.0]
    val = max(4.0, min(9.0, val))
    return round(val * 2) / 2.0

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    bias = np.mean(y_pred - y_true)
    acc_05 = np.mean(np.abs(y_pred - y_true) <= 0.5) * 100.0
    return mae, rmse, bias, acc_05

def print_model_report(name, y_true, y_pred):
    mae, rmse, bias, acc_05 = calculate_metrics(y_true, y_pred)
    print(f"=" * 60)
    print(f"  {name}")
    print(f"=" * 60)
    print(f"  Overall MAE:      {mae:.3f}")
    print(f"  Overall RMSE:     {rmse:.3f}")
    print(f"  Overall Bias:     {bias:+.3f}")
    print(f"  ±0.5 Accuracy:    {acc_05:.1f}%")
    print()

    # Per Group (Low, Mid, High)
    groups = {"Low  (4.0-5.5)": [], "Mid  (6.0-7.0)": [], "High (7.5-9.0)": []}
    for yt, yp in zip(y_true, y_pred):
        grp = get_band_group_label(yt)
        if grp == "Low":
            groups["Low  (4.0-5.5)"].append((yt, yp))
        elif grp == "Mid":
            groups["Mid  (6.0-7.0)"].append((yt, yp))
        else:
            groups["High (7.5-9.0)"].append((yt, yp))

    for g_name, pairs in groups.items():
        if not pairs:
            continue
        g_yt = [p[0] for p in pairs]
        g_yp = [p[1] for p in pairs]
        g_mae, _, g_bias, g_acc05 = calculate_metrics(g_yt, g_yp)
        print(f"  {g_name:<16} | N: {len(pairs):>3} | MAE: {g_mae:.3f} | Bias: {g_bias:+.3f} | ±0.5: {g_acc05:>5.1f}%")

    print()
    print("   Band |    N |   MAE |   Bias |  ±0.5")
    print("  --------------------------------------")
    unique_bands = sorted(list(set(y_true)))
    for b in unique_bands:
        b_pairs = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yt == b]
        b_yt = [p[0] for p in b_pairs]
        b_yp = [p[1] for p in b_pairs]
        b_mae, _, b_bias, b_acc05 = calculate_metrics(b_yt, b_yp)
        print(f"   {b:>4.1f} | {len(b_pairs):>4} | {b_mae:>5.3f} | {b_bias:>+6.3f} | {b_acc05:>4.0f}%")
    print()
    return mae, rmse, bias, acc_05, groups

def main():
    val_path = Path("data/val.csv")
    print(f"Loading validation set from {val_path}...")
    val_data = _load_dataset(val_path)

    # Extract features for all models
    X_a_list, X_f_list, y_true = [], [], []

    print("Extracting features for validation samples...")
    for sample in tqdm(val_data):
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        band = sample.get("band", 5.0)
        if not essay.strip() or not prompt.strip():
            continue
        raw_feat = extract_all_features(essay, prompt)
        norm_feat = normalize_features(raw_feat)
        
        arr_f = features_to_array(norm_feat, FEATURE_KEYS)
        arr_a = features_to_array(norm_feat, FEATURE_KEYS_A)
        
        X_f_list.append(arr_f)
        X_a_list.append(arr_a)
        y_true.append(band)

    X_a = np.vstack(X_a_list)
    X_f = np.vstack(X_f_list)
    y_true = np.array(y_true)

    # 1. Load Model A
    model_a = IELTSModel.load("data/model_exp_A.pkl")
    pred_a_raw = model_a.models["overall"].predict(X_a)
    pred_a = np.array([round_to_ielts_band(p) for p in pred_a_raw])

    # 2. Load Model F
    model_f = IELTSModel.load("data/model_exp_F.pkl")
    pred_f_raw = model_f.models["overall"].predict(X_f)
    pred_f = np.array([round_to_ielts_band(p) for p in pred_f_raw])

    # 3. Load Phase 6F Models (Classifier + 3 Regressors)
    with open("data/model_6f_clf.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("data/model_6f_reg_low.pkl", "rb") as f:
        reg_low = pickle.load(f)
    with open("data/model_6f_reg_mid.pkl", "rb") as f:
        reg_mid = pickle.load(f)
    with open("data/model_6f_reg_high.pkl", "rb") as f:
        reg_high = pickle.load(f)

    # Classifier predictions
    clf_probs = clf.predict_proba(X_f)  # [N, 3] probabilities: LOW, MID, HIGH
    clf_preds = clf.predict(X_f)        # [N] class indices: 0, 1, 2

    # Regressor predictions for all samples
    pred_reg_low = reg_low.predict(X_f)
    pred_reg_mid = reg_mid.predict(X_f)
    pred_reg_high = reg_high.predict(X_f)

    # F1 — Hard Routing
    pred_f1_raw = np.zeros(len(y_true))
    for i in range(len(y_true)):
        grp = clf_preds[i]
        if grp == 0:
            pred_f1_raw[i] = pred_reg_low[i]
        elif grp == 1:
            pred_f1_raw[i] = pred_reg_mid[i]
        else:
            pred_f1_raw[i] = pred_reg_high[i]
    pred_f1 = np.array([round_to_ielts_band(p) for p in pred_f1_raw])

    # F2 — Soft Routing (Probability Weighted Average)
    pred_f2_raw = (clf_probs[:, 0] * pred_reg_low + 
                   clf_probs[:, 1] * pred_reg_mid + 
                   clf_probs[:, 2] * pred_reg_high)
    pred_f2 = np.array([round_to_ielts_band(p) for p in pred_f2_raw])

    # Print reports
    res_a = print_model_report("Model A - Baseline (21 features)", y_true, pred_a)
    res_f = print_model_report("Model F - Enhanced (27 features)", y_true, pred_f)
    res_f1 = print_model_report("Model F1 - Phase 6F (Hard Routing)", y_true, pred_f1)
    res_f2 = print_model_report("Model F2 - Phase 6F (Soft Routing)", y_true, pred_f2)

    # Classifier Confusion Matrix & Accuracy on Val
    y_val_groups = np.array([band_to_group(b) for b in y_true])
    clf_val_acc = np.mean(clf_preds == y_val_groups) * 100.0
    print("=" * 60)
    print("  CLASSIFIER PERFORMANCE ON VAL")
    print("=" * 60)
    print(f"  Validation Group Accuracy: {clf_val_acc:.1f}%\n")
    print("  Group Confusion Matrix (Rows=True, Cols=Pred):")
    print("                Pred LOW  Pred MID  Pred HIGH")
    for true_g in range(3):
        g_label = ["True LOW ", "True MID ", "True HIGH"][true_g]
        row_counts = [np.sum((y_val_groups == true_g) & (clf_preds == pred_g)) for pred_g in range(3)]
        print(f"  {g_label}   {row_counts[0]:>8}  {row_counts[1]:>8}  {row_counts[2]:>8}")
    print()

    # Grand Comparison Table
    print("=" * 70)
    print("  GRAND MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<18} | {'Model A':>9} | {'Model F':>9} | {'F1 (Hard)':>9} | {'F2 (Soft)':>9}")
    print("  " + "-" * 66)
    print(f"  {'Overall MAE':<18} | {res_a[0]:>9.3f} | {res_f[0]:>9.3f} | {res_f1[0]:>9.3f} | {res_f2[0]:>9.3f}")
    print(f"  {'Overall RMSE':<18} | {res_a[1]:>9.3f} | {res_f[1]:>9.3f} | {res_f1[1]:>9.3f} | {res_f2[1]:>9.3f}")
    print(f"  {'Overall Bias':<18} | {res_a[2]:>+9.3f} | {res_f[2]:>+9.3f} | {res_f1[2]:>+9.3f} | {res_f2[2]:>+9.3f}")
    print(f"  {'±0.5 Accuracy':<18} | {res_a[3]:>8.1f}% | {res_f[3]:>8.1f}% | {res_f1[3]:>8.1f}% | {res_f2[3]:>8.1f}%")
    print("  " + "-" * 66)
    
    # Extract Group MAE & Bias
    low_a, mid_a, high_a = res_a[4]["Low  (4.0-5.5)"], res_a[4]["Mid  (6.0-7.0)"], res_a[4]["High (7.5-9.0)"]
    low_f, mid_f, high_f = res_f[4]["Low  (4.0-5.5)"], res_f[4]["Mid  (6.0-7.0)"], res_f[4]["High (7.5-9.0)"]
    low_f1, mid_f1, high_f1 = res_f1[4]["Low  (4.0-5.5)"], res_f1[4]["Mid  (6.0-7.0)"], res_f1[4]["High (7.5-9.0)"]
    low_f2, mid_f2, high_f2 = res_f2[4]["Low  (4.0-5.5)"], res_f2[4]["Mid  (6.0-7.0)"], res_f2[4]["High (7.5-9.0)"]

    print(f"  {'Low Bias':<18} | {calculate_metrics([p[0] for p in low_a], [p[1] for p in low_a])[2]:>+9.3f} | {calculate_metrics([p[0] for p in low_f], [p[1] for p in low_f])[2]:>+9.3f} | {calculate_metrics([p[0] for p in low_f1], [p[1] for p in low_f1])[2]:>+9.3f} | {calculate_metrics([p[0] for p in low_f2], [p[1] for p in low_f2])[2]:>+9.3f}")
    print(f"  {'Low ±0.5 Acc':<18} | {calculate_metrics([p[0] for p in low_a], [p[1] for p in low_a])[3]:>8.1f}% | {calculate_metrics([p[0] for p in low_f], [p[1] for p in low_f])[3]:>8.1f}% | {calculate_metrics([p[0] for p in low_f1], [p[1] for p in low_f1])[3]:>8.1f}% | {calculate_metrics([p[0] for p in low_f2], [p[1] for p in low_f2])[3]:>8.1f}%")
    print("  " + "-" * 66)
    print(f"  {'Mid Bias':<18} | {calculate_metrics([p[0] for p in mid_a], [p[1] for p in mid_a])[2]:>+9.3f} | {calculate_metrics([p[0] for p in mid_f], [p[1] for p in mid_f])[2]:>+9.3f} | {calculate_metrics([p[0] for p in mid_f1], [p[1] for p in mid_f1])[2]:>+9.3f} | {calculate_metrics([p[0] for p in mid_f2], [p[1] for p in mid_f2])[2]:>+9.3f}")
    print(f"  {'Mid ±0.5 Acc':<18} | {calculate_metrics([p[0] for p in mid_a], [p[1] for p in mid_a])[3]:>8.1f}% | {calculate_metrics([p[0] for p in mid_f], [p[1] for p in mid_f])[3]:>8.1f}% | {calculate_metrics([p[0] for p in mid_f1], [p[1] for p in mid_f1])[3]:>8.1f}% | {calculate_metrics([p[0] for p in mid_f2], [p[1] for p in mid_f2])[3]:>8.1f}%")
    print("  " + "-" * 66)
    print(f"  {'High Bias':<18} | {calculate_metrics([p[0] for p in high_a], [p[1] for p in high_a])[2]:>+9.3f} | {calculate_metrics([p[0] for p in high_f], [p[1] for p in high_f])[2]:>+9.3f} | {calculate_metrics([p[0] for p in high_f1], [p[1] for p in high_f1])[2]:>+9.3f} | {calculate_metrics([p[0] for p in high_f2], [p[1] for p in high_f2])[2]:>+9.3f}")
    print(f"  {'High ±0.5 Acc':<18} | {calculate_metrics([p[0] for p in high_a], [p[1] for p in high_a])[3]:>8.1f}% | {calculate_metrics([p[0] for p in high_f], [p[1] for p in high_f])[3]:>8.1f}% | {calculate_metrics([p[0] for p in high_f1], [p[1] for p in high_f1])[3]:>8.1f}% | {calculate_metrics([p[0] for p in high_f2], [p[1] for p in high_f2])[3]:>8.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
