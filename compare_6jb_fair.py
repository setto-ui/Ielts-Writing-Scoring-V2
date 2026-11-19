"""
compare_6jb_fair.py – Fair comparison of Model F_6j (27 feat) vs Model 6J-B (31 feat).
Evaluates models on val.csv (190 samples) across overall, Task 2 only, band subgroups, and feature importance.
STRICT RULE: test.csv is LOCKED.
"""

import numpy as np
import pandas as pd
from pathlib import Path

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


def summarize(model_name, true_bands, pred_bands):
    m_all = compute_metrics(true_bands, pred_bands)
    low_idx  = [i for i, b in enumerate(true_bands) if 4.0 <= b <= 5.5]
    mid_idx  = [i for i, b in enumerate(true_bands) if 6.0 <= b <= 7.0]
    high_idx = [i for i, b in enumerate(true_bands) if 7.5 <= b <= 9.0]
    low_m  = compute_metrics([true_bands[i] for i in low_idx], [pred_bands[i] for i in low_idx]) if low_idx else {}
    mid_m  = compute_metrics([true_bands[i] for i in mid_idx], [pred_bands[i] for i in mid_idx]) if mid_idx else {}
    high_m = compute_metrics([true_bands[i] for i in high_idx], [pred_bands[i] for i in high_idx]) if high_idx else {}
    return {
        "Model": model_name,
        "N": len(true_bands),
        "MAE": f"{m_all['mae']:.3f}",
        "RMSE": f"{m_all['rmse']:.3f}",
        "±0.5": f"{m_all['acc05']:.1f}%",
        "Low Bias": f"{low_m.get('bias', 0.0):>+6.3f}",
        "Mid Bias": f"{mid_m.get('bias', 0.0):>+6.3f}",
        "High Bias": f"{high_m.get('bias', 0.0):>+6.3f}",
        "High MAE": f"{high_m.get('mae', 0.0):.3f}",
        "High ±0.5": f"{high_m.get('acc05', 0.0):.1f}%",
    }


def evaluate_model(model: IELTSModel, dataset: list):
    true_bands, pred_bands, task_types = [], [], []
    feat_keys = model.feature_names
    for sample in dataset:
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        band = float(sample.get("band", 5.0))
        if not essay.strip() or not prompt.strip():
            continue
        raw_feat = extract_all_features(essay, prompt)
        norm_feat = normalize_features(raw_feat)
        arr = features_to_array(norm_feat, feat_keys).reshape(1, -1)
        raw_pred = float(model.predict(arr)["overall"])
        pred = round_to_ielts_band(raw_pred)
        wc = raw_feat.get("word_count", 0)
        true_bands.append(band)
        pred_bands.append(pred)
        task_types.append("Task 2" if wc >= 200 else "Task 1")
    return true_bands, pred_bands, task_types


def main():
    print("=" * 90)
    print("  PHASE 6J-B FAIR COMPARISON (Model F_6j Baseline vs Model 6J-B Candidate)")
    print("=" * 90)

    val_data = _load_dataset(Path("data/val.csv"))
    print(f"Loaded {len(val_data)} validation samples.\n")

    # Load models
    model_a   = IELTSModel.load("data/model_exp_A_6j.pkl")
    model_f   = IELTSModel.load("data/model_exp_F_6j.pkl")
    model_6jb = IELTSModel.load("data/model_exp_6JB.pkl")

    print(f"Model A features:   {len(model_a.feature_names)}")
    print(f"Model F_6j features:{len(model_f.feature_names)}")
    print(f"Model 6J-B features: {len(model_6jb.feature_names)}\n")

    print("Evaluating Model A (21 feat)...")
    tb_a, pb_a, tt_a = evaluate_model(model_a, val_data)

    print("Evaluating Model F_6j (27 feat baseline)...")
    tb_f, pb_f, tt_f = evaluate_model(model_f, val_data)

    print("Evaluating Model 6J-B (31 feat candidate)...")
    tb_6, pb_6, tt_6 = evaluate_model(model_6jb, val_data)

    # 1. Overall Mixed Validation
    rows_mixed = [
        summarize("Model A (21 feat)", tb_a, pb_a),
        summarize("Model F_6j (27 feat)", tb_f, pb_f),
        summarize("Model 6J-B (31 feat)", tb_6, pb_6),
    ]
    df = pd.DataFrame(rows_mixed)
    print("\n1. OVERALL MIXED VALIDATION (N=190, Rounded Bands):")
    print("-" * 105)
    print(df.to_string(index=False))
    print("-" * 105)

    # 2. Task 2 Only
    t2_a  = [(t, p) for t, p, ty in zip(tb_a, pb_a, tt_a) if ty == "Task 2"]
    t2_f  = [(t, p) for t, p, ty in zip(tb_f, pb_f, tt_f) if ty == "Task 2"]
    t2_6  = [(t, p) for t, p, ty in zip(tb_6, pb_6, tt_6) if ty == "Task 2"]

    rows_t2 = [
        summarize("Model A (21 feat)",   [x[0] for x in t2_a], [x[1] for x in t2_a]),
        summarize("Model F_6j (27 feat)",[x[0] for x in t2_f], [x[1] for x in t2_f]),
        summarize("Model 6J-B (31 feat)",[x[0] for x in t2_6], [x[1] for x in t2_6]),
    ]
    df_t2 = pd.DataFrame(rows_t2)
    print(f"\n2. TASK 2 ONLY VALIDATION (N={len(t2_a)}, Rounded Bands):")
    print("-" * 105)
    print(df_t2.to_string(index=False))
    print("-" * 105)

    # 3. Feature Importance for 6J-B
    print("\n3. MODEL 6J-B FEATURE IMPORTANCE (TOP 20 BY GAIN):")
    print("-" * 80)
    booster = model_6jb.models["overall"].booster_
    gains  = booster.feature_importance(importance_type="gain")
    splits = booster.feature_importance(importance_type="split")
    lexical_feats = {"lexical_diversity", "lexical_density", "repetition_rate", "academic_word_ratio"}
    imp_df = pd.DataFrame({
        "Feature": model_6jb.feature_names, "Gain": gains, "Split": splits
    }).sort_values("Gain", ascending=False)

    print(f"{'Rank':<5} | {'Feature':<28} | {'Gain':>10} | {'Split':>6} | {'6J-B Feature?'}")
    print("-" * 75)
    for rank, row in enumerate(imp_df.itertuples(), 1):
        if rank > 20:
            break
        flag = "LEXICAL *" if row.Feature in lexical_feats else ""
        print(f"{rank:<5} | {row.Feature:<28} | {row.Gain:>10.2f} | {row.Split:>6} | {flag}")
    print("-" * 75)

    # 4. Delta summary
    m_f  = compute_metrics(tb_f, pb_f)
    m_6  = compute_metrics(tb_6, pb_6)
    high_f = compute_metrics([tb_f[i] for i, b in enumerate(tb_f) if b >= 7.5],
                              [pb_f[i] for i, b in enumerate(tb_f) if b >= 7.5])
    high_6 = compute_metrics([tb_6[i] for i, b in enumerate(tb_6) if b >= 7.5],
                              [pb_6[i] for i, b in enumerate(tb_6) if b >= 7.5])

    print("\n4. DELTA SUMMARY (Model 6J-B vs Model F_6j Baseline):")
    print(f"  {'Metric':<20} | {'Baseline (F_6j)':>15} | {'Candidate (6J-B)':>16} | {'Delta':>10}")
    print("  " + "-" * 68)
    print(f"  {'MAE (Overall)':<20} | {m_f['mae']:>15.3f} | {m_6['mae']:>16.3f} | {m_6['mae']-m_f['mae']:>+10.3f}")
    print(f"  {'RMSE (Overall)':<20} | {m_f['rmse']:>15.3f} | {m_6['rmse']:>16.3f} | {m_6['rmse']-m_f['rmse']:>+10.3f}")
    print(f"  {'±0.5 Acc (Overall)':<20} | {m_f['acc05']:>14.1f}% | {m_6['acc05']:>15.1f}% | {m_6['acc05']-m_f['acc05']:>+9.1f}%")
    print(f"  {'High Bias (>=7.5)':<20} | {high_f['bias']:>+15.3f} | {high_6['bias']:>+16.3f} | {high_6['bias']-high_f['bias']:>+10.3f}")
    print(f"  {'High MAE (>=7.5)':<20} | {high_f['mae']:>15.3f} | {high_6['mae']:>16.3f} | {high_6['mae']-high_f['mae']:>+10.3f}")
    print(f"  {'High ±0.5 (>=7.5)':<20} | {high_f['acc05']:>14.1f}% | {high_6['acc05']:>15.1f}% | {high_6['acc05']-high_f['acc05']:>+9.1f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
