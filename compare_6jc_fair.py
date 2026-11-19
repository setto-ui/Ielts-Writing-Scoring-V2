"""
compare_6jc_fair.py – Fair comparison of Model 6J-B (31 feat baseline) vs Model 6J-C (34 feat candidate).
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
    print("  PHASE 6J-C FAIR COMPARISON (Model 6J-B Baseline vs Model 6J-C Candidate)")
    print("=" * 90)

    val_data = _load_dataset(Path("data/val.csv"))
    print(f"Loaded {len(val_data)} validation samples.\n")

    # Load models
    model_6jb = IELTSModel.load("data/model_exp_6JB.pkl")
    model_6jc = IELTSModel.load("data/model_exp_6JC.pkl")

    print(f"Model 6J-B baseline features: {len(model_6jb.feature_names)}")
    print(f"Model 6J-C candidate features: {len(model_6jc.feature_names)}\n")

    print("Evaluating Model 6J-B Baseline (31 feat)...")
    tb_b, pb_b, tt_b = evaluate_model(model_6jb, val_data)

    print("Evaluating Model 6J-C Candidate (34 feat)...")
    tb_c, pb_c, tt_c = evaluate_model(model_6jc, val_data)

    # 1. Overall Mixed Validation
    rows_mixed = [
        summarize("Model 6J-B Baseline (31 feat)", tb_b, pb_b),
        summarize("Model 6J-C Candidate (34 feat)", tb_c, pb_c),
    ]
    df = pd.DataFrame(rows_mixed)
    print("\n1. OVERALL MIXED VALIDATION (N=190, Rounded Bands):")
    print("-" * 105)
    print(df.to_string(index=False))
    print("-" * 105)

    # 2. Task 2 Only
    t2_b  = [(t, p) for t, p, ty in zip(tb_b, pb_b, tt_b) if ty == "Task 2"]
    t2_c  = [(t, p) for t, p, ty in zip(tb_c, pb_c, tt_c) if ty == "Task 2"]

    rows_t2 = [
        summarize("Model 6J-B Baseline (31 feat)", [x[0] for x in t2_b], [x[1] for x in t2_b]),
        summarize("Model 6J-C Candidate (34 feat)", [x[0] for x in t2_c], [x[1] for x in t2_c]),
    ]
    df_t2 = pd.DataFrame(rows_t2)
    print(f"\n2. TASK 2 ONLY VALIDATION (N={len(t2_b)}, Rounded Bands):")
    print("-" * 105)
    print(df_t2.to_string(index=False))
    print("-" * 105)

    # 3. Feature Importance for 6J-C
    print("\n3. MODEL 6J-C FEATURE IMPORTANCE (TOP 20 BY GAIN):")
    print("-" * 80)
    booster = model_6jc.models["overall"].booster_
    gains  = booster.feature_importance(importance_type="gain")
    splits = booster.feature_importance(importance_type="split")
    cohesion_feats = {"connective_density", "paragraph_structure_balance", "lexical_overlap"}
    imp_df = pd.DataFrame({
        "Feature": model_6jc.feature_names, "Gain": gains, "Split": splits
    }).sort_values("Gain", ascending=False)

    print(f"{'Rank':<5} | {'Feature':<28} | {'Gain':>10} | {'Split':>6} | {'6J-C Feature?'}")
    print("-" * 75)
    for rank, row in enumerate(imp_df.itertuples(), 1):
        if rank > 20:
            break
        flag = "COHESION *" if row.Feature in cohesion_feats else ""
        print(f"{rank:<5} | {row.Feature:<28} | {row.Gain:>10.2f} | {row.Split:>6} | {flag}")
    print("-" * 75)

    # 4. Delta summary
    m_b = compute_metrics(tb_b, pb_b)
    m_c = compute_metrics(tb_c, pb_c)
    high_b = compute_metrics([tb_b[i] for i, b in enumerate(tb_b) if b >= 7.5],
                              [pb_b[i] for i, b in enumerate(tb_b) if b >= 7.5])
    high_c = compute_metrics([tb_c[i] for i, b in enumerate(tb_c) if b >= 7.5],
                              [pb_c[i] for i, b in enumerate(tb_c) if b >= 7.5])

    print("\n4. DELTA SUMMARY (Model 6J-C vs Model 6J-B Baseline):")
    print(f"  {'Metric':<20} | {'Baseline (6J-B)':>15} | {'Candidate (6J-C)':>16} | {'Delta':>10}")
    print("  " + "-" * 68)
    print(f"  {'MAE (Overall)':<20} | {m_b['mae']:>15.3f} | {m_c['mae']:>16.3f} | {m_c['mae']-m_b['mae']:>+10.3f}")
    print(f"  {'RMSE (Overall)':<20} | {m_b['rmse']:>15.3f} | {m_c['rmse']:>16.3f} | {m_c['rmse']-m_b['rmse']:>+10.3f}")
    print(f"  {'±0.5 Acc (Overall)':<20} | {m_b['acc05']:>14.1f}% | {m_c['acc05']:>15.1f}% | {m_c['acc05']-m_b['acc05']:>+9.1f}%")
    print(f"  {'High Bias (>=7.5)':<20} | {high_b['bias']:>+15.3f} | {high_c['bias']:>+16.3f} | {high_c['bias']-high_b['bias']:>+10.3f}")
    print(f"  {'High MAE (>=7.5)':<20} | {high_b['mae']:>15.3f} | {high_c['mae']:>16.3f} | {high_c['mae']-high_b['mae']:>+10.3f}")
    print(f"  {'High ±0.5 (>=7.5)':<20} | {high_b['acc05']:>14.1f}% | {high_c['acc05']:>15.1f}% | {high_c['acc05']-high_b['acc05']:>+9.1f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
