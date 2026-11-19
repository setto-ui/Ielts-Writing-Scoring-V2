"""
compare_6ja_vs_f_vs_a.py — Compare Phase 6J-A (31 features) vs Model F (27 features) vs Model A (21 features)
Evaluates on val.csv (190 samples total, and 104 Task 2 sub-validation).
STRICT RULE: test.csv is LOCKED.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from scorer.ml_model import IELTSModel
from evaluate import _load_dataset, extract_all_features, features_to_array
from scorer.normalizer import normalize_features
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

def print_group_eval(model_name, true_bands, pred_bands):
    m_all = compute_metrics(true_bands, pred_bands)

    low_idx  = [i for i, b in enumerate(true_bands) if 4.0 <= b <= 5.5]
    mid_idx  = [i for i, b in enumerate(true_bands) if 6.0 <= b <= 7.0]
    high_idx = [i for i, b in enumerate(true_bands) if 7.5 <= b <= 9.0]

    low_m  = compute_metrics([true_bands[i] for i in low_idx], [pred_bands[i] for i in low_idx]) if low_idx else {}
    mid_m  = compute_metrics([true_bands[i] for i in mid_idx], [pred_bands[i] for i in mid_idx]) if mid_idx else {}
    high_m = compute_metrics([true_bands[i] for i in high_idx], [pred_bands[i] for i in high_idx]) if high_idx else {}

    return {
        "Model": model_name,
        "MAE": f"{m_all['mae']:.3f}",
        "RMSE": f"{m_all['rmse']:.3f}",
        "±0.5": f"{m_all['acc05']:.1f}%",
        "Low Bias": f"{low_m.get('bias', 0.0):>+6.3f}",
        "Mid Bias": f"{mid_m.get('bias', 0.0):>+6.3f}",
        "High Bias": f"{high_m.get('bias', 0.0):>+6.3f}",
        "High ±0.5": f"{high_m.get('acc05', 0.0):.1f}%",
    }

def evaluate_model_on_dataset(model: IELTSModel, dataset: list, rounded: bool = True):
    true_bands = []
    pred_bands = []
    task_types = []

    # Use model's own feature_names to guarantee exact feature order & match
    feature_names = model.feature_names

    for sample in dataset:
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        band = float(sample.get("band", 5.0))
        if not essay.strip() or not prompt.strip():
            continue

        raw_feat = extract_all_features(essay, prompt)
        norm_feat = normalize_features(raw_feat)

        wc = raw_feat.get("word_count", 0)
        is_task2 = wc >= 200

        arr = features_to_array(norm_feat, feature_names).reshape(1, -1)
        raw_pred = float(model.predict(arr)["overall"])

        pred = round_to_ielts_band(raw_pred) if rounded else raw_pred

        true_bands.append(band)
        pred_bands.append(pred)
        task_types.append("Task 2" if is_task2 else "Task 1")

    return true_bands, pred_bands, task_types

def main():
    print("=" * 85)
    print("  PHASE 6J-A MODEL EVALUATION & COMPARISON")
    print("=" * 85)

    val_path = Path("data/val.csv")
    val_data = _load_dataset(val_path)
    print(f"Loaded {len(val_data)} validation samples from {val_path}.\n")

    model_a = IELTSModel.load("data/model_exp_A.pkl")
    model_f = IELTSModel.load("data/model_exp_F.pkl")
    model_6ja = IELTSModel.load("data/model_exp_6JA.pkl")

    print("Extracting features and predicting for Model A (21 feat)...")
    tb_a, pb_a, tt_a = evaluate_model_on_dataset(model_a, val_data, rounded=True)

    print("Extracting features and predicting for Model F (27 feat)...")
    tb_f, pb_f, tt_f = evaluate_model_on_dataset(model_f, val_data, rounded=True)

    print("Extracting features and predicting for Model 6J-A (31 feat)...")
    tb_6ja, pb_6ja, tt_6ja = evaluate_model_on_dataset(model_6ja, val_data, rounded=True)

    # 1. Mixed Validation (190 samples)
    res_a_mixed = print_group_eval("Model A (21 feat)", tb_a, pb_a)
    res_f_mixed = print_group_eval("Model F (27 feat)", tb_f, pb_f)
    res_6ja_mixed = print_group_eval("Model 6J-A (31 feat)", tb_6ja, pb_6ja)

    df_mixed = pd.DataFrame([res_a_mixed, res_f_mixed, res_6ja_mixed])

    print("\n1. OVERALL MIXED VALIDATION (N=190 samples, Rounded Bands):")
    print("-" * 85)
    print(df_mixed.to_string(index=False))
    print("-" * 85)

    # 2. Task 2 Validation Only (104 samples)
    t2_idx = [i for i, t in enumerate(tt_a) if t == "Task 2"]
    t2_tb = [tb_a[i] for i in t2_idx]

    res_a_t2 = print_group_eval("Model A (21 feat)", t2_tb, [pb_a[i] for i in t2_idx])
    res_f_t2 = print_group_eval("Model F (27 feat)", t2_tb, [pb_f[i] for i in t2_idx])
    res_6ja_t2 = print_group_eval("Model 6J-A (31 feat)", t2_tb, [pb_6ja[i] for i in t2_idx])

    df_t2 = pd.DataFrame([res_a_t2, res_f_t2, res_6ja_t2])

    print(f"\n2. TASK 2 ONLY VALIDATION (N={len(t2_idx)} samples, Rounded Bands):")
    print("-" * 85)
    print(df_t2.to_string(index=False))
    print("-" * 85)

    # 3. Feature Importance for Model 6J-A
    print("\n3. MODEL 6J-A FEATURE IMPORTANCE (TOP 15 BY GAIN):")
    print("-" * 85)
    if hasattr(model_6ja.models["overall"], "booster_"):
        booster = model_6ja.models["overall"].booster_
        gains = booster.feature_importance(importance_type="gain")
        splits = booster.feature_importance(importance_type="split")
        imp_df = pd.DataFrame({
            "Feature": model_6ja.feature_names,
            "Gain": gains,
            "Split": splits,
        }).sort_values(by="Gain", ascending=False)

        print(f"{'Rank':<5} | {'Feature':<28} | {'Gain':>10} | {'Split':>6} | {'New 6J-A?':<8}")
        print("-" * 75)
        new_feats = {"causal_relation_density", "example_density", "claim_support_ratio", "task2_length_shortfall"}
        for rank, row in enumerate(imp_df.head(15).itertuples(), start=1):
            is_new = "YES *" if row.Feature in new_feats else ""
            print(f"{rank:<5} | {row.Feature:<28} | {row.Gain:>10.2f} | {row.Split:>6} | {is_new:<8}")
        print("-" * 75)

    print("=" * 85)

if __name__ == "__main__":
    main()
