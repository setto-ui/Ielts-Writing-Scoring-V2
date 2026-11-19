"""
feature_stats_6j_b.py – Diagnostic profiling of Phase 6J-B Lexical Quality features.
Audits features across train.csv + val.csv (1,076 samples).
Computes Pearson r, Spearman r, mean/std, zero rate, unique count, band means, and monotonicity.
STRICT RULE: Does NOT auto-drop features. Keeps all 4 features for candidate model evaluation.
STRICT RULE: test.csv is LOCKED.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

from evaluate import _load_dataset
from scorer.features_lexical import extract_lexical_features

NEW_FEATURES = [
    "lexical_diversity",
    "lexical_density",
    "repetition_rate",
    "academic_word_ratio",
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
    print("  PHASE 6J-B LEXICAL QUALITY FEATURE AUDIT (1,076 SAMPLES)")
    print("=" * 90)

    train_samples = _load_dataset(Path("data/train.csv"))
    val_samples = _load_dataset(Path("data/val.csv"))
    dataset = train_samples + val_samples
    print(f"Loaded {len(dataset)} samples (train: {len(train_samples)}, val: {len(val_samples)}).\n")

    records = []
    for sample in tqdm(dataset, desc="Extracting 6J-B features"):
        essay = sample.get("essay", "")
        band = float(sample.get("band", 5.0))
        if not essay.strip():
            continue
        feat = extract_lexical_features(essay)
        feat["band"] = band
        records.append(feat)

    df = pd.DataFrame(records)

    summary_rows = []
    print("\n--- 1. FEATURE DIAGNOSTIC METRICS ---")
    print("-" * 90)
    print(f"{'Feature':<22} | {'Pearson r':>9} | {'Spearman r':>10} | {'Mean ± Std':>16} | {'Zero %':>7} | {'Unique':>6} | {'Monotonic?'}")
    print("-" * 90)

    for feat in NEW_FEATURES:
        vals = df[feat].values
        bands = df["band"].values

        p_r, _ = pearsonr(vals, bands)
        s_r, _ = spearmanr(vals, bands)

        mean_val = np.mean(vals)
        std_val = np.std(vals)
        zero_pct = np.mean(vals == 0.0) * 100.0
        unique_cnt = len(np.unique(vals))

        # Band means for monotonicity
        b_means = []
        for low, high in BAND_BUCKETS:
            subset = df[(df["band"] >= low) & (df["band"] <= high)][feat]
            b_means.append(subset.mean() if len(subset) > 0 else 0.0)

        mono_str = check_monotonicity(b_means)

        print(f"{feat:<22} | {p_r:>+9.4f} | {s_r:>+10.4f} | {mean_val:6.3f} ± {std_val:5.3f} | {zero_pct:>6.1f}% | {unique_cnt:>6} | {mono_str}")

        summary_rows.append({
            "Feature": feat,
            "Pearson_r": round(p_r, 4),
            "Spearman_r": round(s_r, 4),
            "Mean": round(mean_val, 4),
            "Std": round(std_val, 4),
            "Zero_Pct": round(zero_pct, 2),
            "Unique": unique_cnt,
            "Monotonic": mono_str,
        })

    print("-" * 90)

    print("\n--- 2. BAND BUCKET MEANS ---")
    print("-" * 90)
    header = f"{'Feature':<22} | " + " | ".join([f"{lbl:>18}" for lbl in BAND_LABELS])
    print(header)
    print("-" * 90)

    for feat in NEW_FEATURES:
        b_means_str = []
        for low, high in BAND_BUCKETS:
            subset = df[(df["band"] >= low) & (df["band"] <= high)][feat]
            mean_v = subset.mean() if len(subset) > 0 else 0.0
            b_means_str.append(f"{mean_v:>18.4f}")
        print(f"{feat:<22} | " + " | ".join(b_means_str))

    print("-" * 90)
    print("\nDiagnostic profiling complete. All 4 features retained for Model 6J-B training.")


if __name__ == "__main__":
    main()
