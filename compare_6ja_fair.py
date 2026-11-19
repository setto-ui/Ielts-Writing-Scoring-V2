"""
compare_6ja_fair.py — Fair comparison of Model A vs F vs 6J-A
Uses retrained baselines (model_exp_A_6j.pkl, model_exp_F_6j.pkl) and model_exp_6JA.pkl.
All three trained on the same updated feature pipeline (post-6J-A changes).
STRICT RULE: test.csv is LOCKED.
"""
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
    print("  PHASE 6J-A FAIR COMPARISON (All models retrained on updated pipeline)")
    print("=" * 90)

    val_data = _load_dataset(Path("data/val.csv"))
    print(f"Loaded {len(val_data)} validation samples.\n")

    # Load models
    model_a   = IELTSModel.load("data/model_exp_A_6j.pkl")
    model_f   = IELTSModel.load("data/model_exp_F_6j.pkl")
    model_6ja = IELTSModel.load("data/model_exp_6JA.pkl")

    print(f"Model A features:   {len(model_a.feature_names)}")
    print(f"Model F features:   {len(model_f.feature_names)}")
    print(f"Model 6JA features: {len(model_6ja.feature_names)}\n")

    print("Evaluating Model A (21 feat)...")
    tb_a, pb_a, tt_a = evaluate_model(model_a, val_data)

    print("Evaluating Model F (27 feat)...")
    tb_f, pb_f, tt_f = evaluate_model(model_f, val_data)

    print("Evaluating Model 6J-A (31 feat)...")
    tb_6, pb_6, tt_6 = evaluate_model(model_6ja, val_data)

    # ── 1. Overall Mixed Validation ──────────────────────────────────────────
    rows_mixed = [
        summarize("Model A (21 feat)", tb_a, pb_a),
        summarize("Model F (27 feat)", tb_f, pb_f),
        summarize("Model 6J-A (31 feat)", tb_6, pb_6),
    ]
    df = pd.DataFrame(rows_mixed)
    print("\n1. OVERALL MIXED VALIDATION (N=190, Rounded Bands):")
    print("-" * 90)
    print(df.to_string(index=False))
    print("-" * 90)

    # ── 2. Task 2 Only ───────────────────────────────────────────────────────
    t2_a  = [(t, p) for t, p, ty in zip(tb_a, pb_a, tt_a) if ty == "Task 2"]
    t2_f  = [(t, p) for t, p, ty in zip(tb_f, pb_f, tt_f) if ty == "Task 2"]
    t2_6  = [(t, p) for t, p, ty in zip(tb_6, pb_6, tt_6) if ty == "Task 2"]

    rows_t2 = [
        summarize("Model A (21 feat)",   [x[0] for x in t2_a], [x[1] for x in t2_a]),
        summarize("Model F (27 feat)",   [x[0] for x in t2_f], [x[1] for x in t2_f]),
        summarize("Model 6J-A (31 feat)",[x[0] for x in t2_6], [x[1] for x in t2_6]),
    ]
    df_t2 = pd.DataFrame(rows_t2)
    print(f"\n2. TASK 2 ONLY (N={len(t2_a)}, Rounded Bands):")
    print("-" * 90)
    print(df_t2.to_string(index=False))
    print("-" * 90)

    # ── 3. Feature Importance for 6J-A ───────────────────────────────────────
    print("\n3. MODEL 6J-A FEATURE IMPORTANCE (TOP 20 BY GAIN):")
    print("-" * 80)
    booster = model_6ja.models["overall"].booster_
    gains  = booster.feature_importance(importance_type="gain")
    splits = booster.feature_importance(importance_type="split")
    new_feats = {"causal_relation_density", "example_density", "claim_support_ratio", "task2_length_shortfall"}
    imp_df = pd.DataFrame({
        "Feature": model_6ja.feature_names, "Gain": gains, "Split": splits
    }).sort_values("Gain", ascending=False)

    print(f"{'Rank':<5} | {'Feature':<28} | {'Gain':>10} | {'Split':>6} | {'New?'}")
    print("-" * 70)
    for rank, row in enumerate(imp_df.itertuples(), 1):
        if rank > 20:
            break
        flag = "NEW *" if row.Feature in new_feats else ""
        print(f"{rank:<5} | {row.Feature:<28} | {row.Gain:>10.2f} | {row.Split:>6} | {flag}")
    print("-" * 70)

    # Delta summary
    m_a  = compute_metrics(tb_a, pb_a)
    m_f  = compute_metrics(tb_f, pb_f)
    m_6  = compute_metrics(tb_6, pb_6)
    high_a = compute_metrics([tb_a[i] for i, b in enumerate(tb_a) if b >= 7.5],
                              [pb_a[i] for i, b in enumerate(tb_a) if b >= 7.5])
    high_f = compute_metrics([tb_f[i] for i, b in enumerate(tb_f) if b >= 7.5],
                              [pb_f[i] for i, b in enumerate(tb_f) if b >= 7.5])
    high_6 = compute_metrics([tb_6[i] for i, b in enumerate(tb_6) if b >= 7.5],
                              [pb_6[i] for i, b in enumerate(tb_6) if b >= 7.5])

    print("\n4. DELTA SUMMARY vs MODEL F:")
    print(f"  {'Metric':<18} | {'A vs F':>8} | {'6JA vs F':>8}")
    print("  " + "-" * 42)
    print(f"  {'MAE':<18} | {m_a['mae']-m_f['mae']:>+8.3f} | {m_6['mae']-m_f['mae']:>+8.3f}")
    print(f"  {'RMSE':<18} | {m_a['rmse']-m_f['rmse']:>+8.3f} | {m_6['rmse']-m_f['rmse']:>+8.3f}")
    print(f"  {'±0.5 Acc':<18} | {m_a['acc05']-m_f['acc05']:>+8.1f} | {m_6['acc05']-m_f['acc05']:>+8.1f}")
    print(f"  {'High Bias':<18} | {high_a['bias']-high_f['bias']:>+8.3f} | {high_6['bias']-high_f['bias']:>+8.3f}")
    print(f"  {'High ±0.5':<18} | {high_a['acc05']-high_f['acc05']:>+8.1f} | {high_6['acc05']-high_f['acc05']:>+8.1f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
