"""
run_phase6g_error_analysis.py - Phase 6G: Error Analysis on Validation Set (190 samples) using Model F

Generates CSV files for deep error analysis:
  1. data/error_analysis_val_all.csv: All 190 validation essays with true/pred band, error, and 27 feature values.
  2. data/worst_20_errors.csv: Top 20 worst absolute predictions.
  3. data/high_band_errors.csv: High-band essays (True >= 7.5) sorted by absolute error.
  4. data/low_band_errors.csv: Low-band essays (True <= 5.5) sorted by absolute error.
  5. data/correct_predictions.csv: Accurate predictions (|Error| <= 0.5).

Prints statistical comparison of feature distributions across error buckets to identify missing signals.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from scorer.ml_model import IELTSModel
from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import normalize_features
from scorer.scorer import round_to_ielts_band


def main():
    val_path = Path("data/val.csv")
    print(f"Loading validation set from {val_path}...")
    val_data = _load_dataset(val_path)

    print("Loading Model F candidate (data/model_exp_F.pkl)...")
    model_f = IELTSModel.load("data/model_exp_F.pkl")

    records = []
    print("Running Model F inference and gathering feature statistics...")
    for idx, sample in enumerate(tqdm(val_data)):
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        true_band = float(sample.get("band", 5.0))
        if not essay.strip() or not prompt.strip():
            continue

        raw_feat = extract_all_features(essay, prompt)
        norm_feat = normalize_features(raw_feat)
        X_arr = features_to_array(norm_feat, FEATURE_KEYS).reshape(1, -1)
        
        raw_pred = float(model_f.models["overall"].predict(X_arr)[0])
        pred_band = round_to_ielts_band(raw_pred)
        
        err = pred_band - true_band
        abs_err = abs(err)

        row = {
            "val_id": idx,
            "prompt": prompt,
            "essay": essay,
            "true_band": true_band,
            "raw_pred": raw_pred,
            "pred_band": pred_band,
            "signed_error": err,
            "abs_error": abs_err,
        }
        # Add all 27 feature values (raw)
        for k in FEATURE_KEYS:
            row[f"feat_{k}"] = raw_feat.get(k, 0.0)

        records.append(row)

    df = pd.DataFrame(records)
    print(f"\nExtracted data for {len(df)} valid validation samples.")

    # Sort and create subsets
    df_all = df.sort_values(by="abs_error", ascending=False)
    
    # 1. All validation essays
    df_all.to_csv("data/error_analysis_val_all.csv", index=False)
    print("  Saved data/error_analysis_val_all.csv")

    # 2. Worst 20 errors
    df_worst20 = df_all.head(20)
    df_worst20.to_csv("data/worst_20_errors.csv", index=False)
    print("  Saved data/worst_20_errors.csv")

    # 3. High Band errors (True >= 7.5)
    df_high = df_all[df_all["true_band"] >= 7.5]
    df_high.to_csv("data/high_band_errors.csv", index=False)
    print(f"  Saved data/high_band_errors.csv ({len(df_high)} samples)")

    # 4. Low Band errors (True <= 5.5)
    df_low = df_all[df_all["true_band"] <= 5.5]
    df_low.to_csv("data/low_band_errors.csv", index=False)
    print(f"  Saved data/low_band_errors.csv ({len(df_low)} samples)")

    # 5. Accurate predictions (|Error| <= 0.5)
    df_correct = df_all[df_all["abs_error"] <= 0.5]
    df_correct.to_csv("data/correct_predictions.csv", index=False)
    print(f"  Saved data/correct_predictions.csv ({len(df_correct)} samples)")

    # Summary Statistics across error groups
    print("\n" + "=" * 75)
    print("  PHASE 6G FEATURE MEANS BY ERROR BUCKET")
    print("=" * 75)
    
    buckets = {
        "High Band (>=7.5)": df_high,
        "Low Band (<=5.5)": df_low,
        "Mid Band (6.0-7.0)": df[(df["true_band"] >= 6.0) & (df["true_band"] <= 7.0)],
        "Correct (|Err|<=0.5)": df_correct,
        "Worst 20 Errors": df_worst20,
    }

    # Focus features for diagnosis
    key_diag_features = [
        "word_count",
        "paragraph_count",
        "avg_paragraph_length",
        "avg_sentence_length",
        "sentence_length_std",
        "long_word_ratio",
        "rare_word_ratio",
        "mattr",
        "grammar_error_rate",
        "grammar_error_density",
        "complex_clause_ratio",
        "semantic_coherence",
        "argument_depth_score",
        "task_coverage_score",
        "prompt_keyword_overlap",
    ]

    header_cols = [f"{name:<20}" for name in buckets.keys()]
    print(f"  {'Feature':<25} | " + " | ".join(header_cols))
    print("  " + "-" * 135)

    for feat in key_diag_features:
        feat_col = f"feat_{feat}"
        vals = []
        for b_name, b_df in buckets.items():
            if len(b_df) > 0:
                mean_val = b_df[feat_col].mean()
                vals.append(f"{mean_val:>20.4f}")
            else:
                vals.append(f"{'N/A':>20}")
        print(f"  {feat:<25} | " + " | ".join(vals))
    print("=" * 75)

    # Detailed inspection of Top 5 Worst High-Band Errors
    print("\n" + "=" * 75)
    print("  TOP 5 HIGH-BAND UNDER-SCORED ESSAYS INSPECTION (True >= 7.5)")
    print("=" * 75)
    df_high_worst = df_high.head(5)
    for _, row in df_high_worst.iterrows():
        print(f"  [ID {row['val_id']}] True: {row['true_band']} | Pred: {row['pred_band']} (Raw: {row['raw_pred']:.2f}) | Signed Err: {row['signed_error']:+.2f}")
        print(f"     Prompt: {row['prompt'][:80]}...")
        print(f"     Word count: {row['feat_word_count']:.0f} | Para len: {row['feat_avg_paragraph_length']:.1f} | Long word ratio: {row['feat_long_word_ratio']:.3f} | Grammar err rate: {row['feat_grammar_error_rate']:.3f}")
        print(f"     Semantic Coherence: {row['feat_semantic_coherence']:.3f} | Task Coverage: {row['feat_task_coverage_score']:.3f} | Keyword overlap: {row['feat_prompt_keyword_overlap']:.3f}")
        print("  " + "-" * 75)


if __name__ == "__main__":
    main()
