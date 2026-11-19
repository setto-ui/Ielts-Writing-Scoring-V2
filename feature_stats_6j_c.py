"""
feature_stats_6j_c.py – Diagnostic profiling of Phase 6J-C Cohesion & Structure features.
Audits features across train.csv + val.csv (1,076 samples).
Computes Pearson r, Spearman r, mean/std, min/max, zero rate, unique count, band means, monotonicity,
and cross-feature correlations against existing features (to check for information redundancy).
STRICT RULE: Does NOT auto-drop features. Keeps all candidate features for Model 6J-C evaluation.
STRICT RULE: test.csv is LOCKED.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

from evaluate import _load_dataset, extract_all_features

NEW_FEATURES = [
    "connective_density",
    "paragraph_structure_balance",
    "lexical_overlap",
]

EXISTING_REF_FEATURES = [
    "word_count",
    "paragraph_count",
    "semantic_coherence",
    "lexical_diversity",
    "discourse_marker_count",
]

BAND_BUCKETS = [(4.0, 5.5), (6.0, 6.5), (7.0, 7.5), (8.0, 9.0)]
BAND_LABELS = ["Low (4.0-5.5)", "Mid (6.0-6.5)", "High-Mid (7.0-7.5)", "High (8.0-9.0)"]


def check_monotonicity(means: list[float]) -> str:
    """Check if band means strictly increase or decrease."""
    diffs = np.diff(means)
    if np.all(diffs >= 0):
        return "YES (Increasing)"
    elif np.all(diffs <= 0):
        return "YES (Decreasing)"
    else:
        return "NO (Non-monotonic)"


def main():
    print("=" * 90)
    print("  PHASE 6J-C COHESION & STRUCTURE FEATURE AUDIT (1,076 SAMPLES)")
    print("=" * 90)

    train_samples = _load_dataset(Path("data/train.csv"))
    val_samples = _load_dataset(Path("data/val.csv"))
    dataset = train_samples + val_samples
    print(f"Loaded {len(dataset)} samples (train: {len(train_samples)}, val: {len(val_samples)}).\n")

    records = []
    for sample in tqdm(dataset, desc="Extracting 6J-C features"):
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        band = float(sample.get("band", 5.0))
        if not essay.strip() or not prompt.strip():
            continue
        feat = extract_all_features(essay, prompt)
        feat["band"] = band
        records.append(feat)

    df = pd.DataFrame(records)

    # --- 1. Diagnostic Metrics Table ---
    print("\n--- 1. FEATURE DIAGNOSTIC METRICS ---")
    print("-" * 105)
    print(f"{'Feature':<28} | {'Pearson r':>9} | {'Spearman r':>10} | {'Mean ± Std':>16} | {'Min / Max':>13} | {'Zero %':>7} | {'Unique':>6} | {'Monotonic?'}")
    print("-" * 105)

    for feat in NEW_FEATURES:
        vals = df[feat].values
        bands = df["band"].values

        p_r, _ = pearsonr(vals, bands)
        s_r, _ = spearmanr(vals, bands)

        mean_val = np.mean(vals)
        std_val = np.std(vals)
        min_val = np.min(vals)
        max_val = np.max(vals)
        zero_pct = np.mean(vals == 0.0) * 100.0
        unique_cnt = len(np.unique(vals))

        b_means = []
        for low, high in BAND_BUCKETS:
            subset = df[(df["band"] >= low) & (df["band"] <= high)][feat]
            b_means.append(subset.mean() if len(subset) > 0 else 0.0)

        mono_str = check_monotonicity(b_means)

        print(f"{feat:<28} | {p_r:>+9.4f} | {s_r:>+10.4f} | {mean_val:6.3f} ± {std_val:5.3f} | {min_val:5.2f}/{max_val:5.2f} | {zero_pct:>6.1f}% | {unique_cnt:>6} | {mono_str}")

    print("-" * 105)

    # --- 2. Band Bucket Means ---
    print("\n--- 2. BAND BUCKET MEANS ---")
    print("-" * 105)
    header = f"{'Feature':<28} | " + " | ".join([f"{lbl:>18}" for lbl in BAND_LABELS])
    print(header)
    print("-" * 105)

    for feat in NEW_FEATURES:
        b_means_str = []
        for low, high in BAND_BUCKETS:
            subset = df[(df["band"] >= low) & (df["band"] <= high)][feat]
            mean_v = subset.mean() if len(subset) > 0 else 0.0
            b_means_str.append(f"{mean_v:>18.4f}")
        print(f"{feat:<28} | " + " | ".join(b_means_str))

    print("-" * 105)

    # --- 3. Cross-Feature Correlation Matrix ---
    print("\n--- 3. CROSS-FEATURE CORRELATION MATRIX (Redundancy Audit) ---")
    print("-" * 105)
    header_cross = f"{'6J-C Feature':<28} | " + " | ".join([f"{ref:>20}" for ref in EXISTING_REF_FEATURES])
    print(header_cross)
    print("-" * 105)

    for feat in NEW_FEATURES:
        row_corrs = []
        for ref in EXISTING_REF_FEATURES:
            if ref in df.columns:
                corr_val, _ = pearsonr(df[feat].values, df[ref].values)
                row_corrs.append(f"{corr_val:>+20.4f}")
            else:
                row_corrs.append(f"{'N/A':>20}")
        print(f"{feat:<28} | " + " | ".join(row_corrs))

    print("-" * 105)
    print("\nDiagnostic profiling complete. All 3 candidate features retained for Model 6J-C training.")


if __name__ == "__main__":
    main()
