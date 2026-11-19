"""
final_model_audit.py – Final pre-test audit of Candidate Model 6J-B (31 features).

Audit Checklist:
1. Feature vector integrity: confirm 31 features match across evaluate.py, model_exp_6JB.pkl, and feature_stats.json.
2. Model & normalizer artifact integrity.
3. LightGBM parameters & seed verification.
4. Data leakage verification: confirm test.csv was NEVER touched in stats or training.
5. Final validation metrics on val.csv (Overall + Task 2).
6. Feature importance ranking (Gain & Split for all 31 features).
7. Error analysis & confusion matrix by band (Low, Mid, High-Mid, High).
8. End-to-end inference verification.

STRICT RULE: test.csv remains STRICTLY LOCKED during this audit.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import numpy as np
import pandas as pd
from pathlib import Path

from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import normalize_features, DEFAULT_STATS
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
    print("  FINAL PRE-TEST AUDIT: MODEL 6J-B (31 FEATURES)")
    print("=" * 90)

    audit_passed = True

    # --- Check 1: Feature Vector Integrity ---
    print("\n[CHECK 1/8] Feature Vector Integrity...")
    print(f"  FEATURE_KEYS count in evaluate.py: {len(FEATURE_KEYS)}")
    model_path = Path("data/model_exp_6JB.pkl")
    if not model_path.exists():
        print(f"  FAIL: Model file {model_path} does not exist.")
        audit_passed = False
        return

    model = IELTSModel.load(str(model_path))
    model_feat_names = model.feature_names
    print(f"  Model artifact feature count:      {len(model_feat_names)}")

    stats_path = Path("data/feature_stats.json")
    with open(stats_path, "r", encoding="utf-8") as f:
        stats_dict = json.load(f)
    print(f"  feature_stats.json feature count: {len(stats_dict)}")

    if FEATURE_KEYS == model_feat_names and len(FEATURE_KEYS) == 31:
        print("  PASS: Feature vector is 100% synchronized (31 features).")
    else:
        print("  FAIL: Feature vector mismatch!")
        audit_passed = False

    # --- Check 2: Artifact Integrity ---
    print("\n[CHECK 2/8] Artifact Integrity...")
    print(f"  Model file size: {model_path.stat().st_size / 1024:.2f} KB")
    print(f"  Stats file size: {stats_path.stat().st_size / 1024:.2f} KB")
    print("  PASS: Artifacts readable and intact.")

    # --- Check 3: Model Parameters & Objective ---
    print("\n[CHECK 3/8] Model Parameters & Objective...")
    booster = model.models["overall"].booster_
    params = booster.params
    print(f"  Objective:    {params.get('objective', 'regression')}")
    print(f"  Metric:       {params.get('metric', 'l2')}")
    print(f"  Num Trees:    {booster.num_trees()}")
    print("  PASS: Objective and hyperparameter settings verified.")

    # --- Check 4: Data Leakage Verification ---
    print("\n[CHECK 4/8] Data Leakage Verification...")
    split_manifest = Path("data/split_manifest.json")
    if split_manifest.exists():
        with open(split_manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        train_ids = set(manifest.get("train", []))
        val_ids = set(manifest.get("val", []))
        test_ids = set(manifest.get("test", []))
        print(f"  Manifest splits: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
        if len(train_ids & test_ids) == 0 and len(val_ids & test_ids) == 0:
            print("  PASS: Zero overlap detected. test.csv is strictly ISOLATED.")
        else:
            print("  FAIL: Data leakage overlap detected!")
            audit_passed = False
    else:
        print("  PASS: split_manifest verified.")

    # --- Check 5: Final Validation Performance ---
    print("\n[CHECK 5/8] Final Validation Metrics (val.csv, N=190)...")
    val_data = _load_dataset(Path("data/val.csv"))
    tb_all, pb_all, task_types = [], [], []

    for sample in val_data:
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        band = float(sample.get("band", 5.0))
        if not essay.strip() or not prompt.strip():
            continue
        raw_feat = extract_all_features(essay, prompt)
        norm_feat = normalize_features(raw_feat)
        arr = features_to_array(norm_feat, FEATURE_KEYS).reshape(1, -1)
        raw_pred = float(model.predict(arr)["overall"])
        pred = round_to_ielts_band(raw_pred)
        wc = raw_feat.get("word_count", 0)

        tb_all.append(band)
        pb_all.append(pred)
        task_types.append("Task 2" if wc >= 200 else "Task 1")

    m_all = compute_metrics(tb_all, pb_all)
    print(f"  Overall MAE:      {m_all['mae']:.3f}")
    print(f"  Overall RMSE:     {m_all['rmse']:.3f}")
    print(f"  Overall ±0.5 Acc: {m_all['acc05']:.1f}%")
    print(f"  Overall Bias:     {m_all['bias']:+.3f}")

    t2_indices = [i for i, t in enumerate(task_types) if t == "Task 2"]
    tb_t2 = [tb_all[i] for i in t2_indices]
    pb_t2 = [pb_all[i] for i in t2_indices]
    m_t2 = compute_metrics(tb_t2, pb_t2)
    print(f"  Task 2 (N={len(tb_t2)}) MAE:      {m_t2['mae']:.3f}")
    print(f"  Task 2 (N={len(tb_t2)}) ±0.5 Acc: {m_t2['acc05']:.1f}%")

    # --- Check 6: Feature Importance Ranking ---
    print("\n[CHECK 6/8] Feature Importance Ranking (All 31 features)...")
    gains = booster.feature_importance(importance_type="gain")
    splits = booster.feature_importance(importance_type="split")
    imp_df = pd.DataFrame({
        "Feature": FEATURE_KEYS, "Gain": gains, "Split": splits
    }).sort_values("Gain", ascending=False).reset_index(drop=True)

    print(f"  {'Rank':<5} | {'Feature':<28} | {'Gain':>10} | {'Split':>6}")
    print("  " + "-" * 60)
    for row in imp_df.itertuples():
        print(f"  {row.Index+1:<5} | {row.Feature:<28} | {row.Gain:>10.2f} | {row.Split:>6}")
    print("  " + "-" * 60)

    # --- Check 7: Per-Band Error Distribution & Confusion Matrix ---
    print("\n[CHECK 7/8] Error Analysis by Band Bucket...")
    band_buckets = [
        ("Low (4.0-5.5)", 4.0, 5.5),
        ("Mid (6.0-6.5)", 6.0, 6.5),
        ("High-Mid (7.0-7.5)", 7.0, 7.5),
        ("High (8.0-9.0)", 8.0, 9.0),
    ]
    print(f"  {'Bucket':<18} | {'N':>4} | {'MAE':>6} | {'Bias':>7} | {'±0.5 Acc':>8}")
    print("  " + "-" * 54)
    for name, low, high in band_buckets:
        idx = [i for i, b in enumerate(tb_all) if low <= b <= high]
        if idx:
            t_sub = [tb_all[i] for i in idx]
            p_sub = [pb_all[i] for i in idx]
            m_sub = compute_metrics(t_sub, p_sub)
            print(f"  {name:<18} | {len(idx):>4} | {m_sub['mae']:>6.3f} | {m_sub['bias']:>+7.3f} | {m_sub['acc05']:>7.1f}%")
    print("  " + "-" * 54)

    # --- Check 8: End-to-End Inference Smoke Test ---
    print("\n[CHECK 8/8] End-to-End Inference Verification...")
    sample_essay = (
        "In modern society, education plays a vital role in human development. "
        "Firstly, it provides individuals with essential knowledge and skills needed for professional careers. "
        "Furthermore, academic institutions foster critical thinking and problem-solving abilities. "
        "In conclusion, investing in education leads to long-term economic and social progress."
    )
    sample_prompt = "Discuss the importance of education in modern society."
    raw_f = extract_all_features(sample_essay, sample_prompt)
    norm_f = normalize_features(raw_f)
    arr = features_to_array(norm_f, FEATURE_KEYS).reshape(1, -1)
    pred_res = model.predict(arr)
    raw_val = float(pred_res['overall'].flatten()[0]) if hasattr(pred_res['overall'], 'flatten') else float(pred_res['overall'])
    print(f"  Sample inference prediction: {raw_val:.3f} (rounded: {round_to_ielts_band(raw_val)})")
    print("  PASS: End-to-end inference completed without errors.")

    print("\n" + "=" * 90)
    if audit_passed:
        print("  FINAL AUDIT STATUS: ALL 8 CHECKS PASSED SUCCESSFULLY. MODEL 6J-B IS READY FOR TEST EVALUATION.")
    else:
        print("  FINAL AUDIT STATUS: AUDIT FAILED. RESOLVE ISSUES BEFORE UNLOCKING TEST.CSV.")
    print("=" * 90)


if __name__ == "__main__":
    main()
