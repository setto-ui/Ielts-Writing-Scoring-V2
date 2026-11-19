"""
feature_stats_6j_a.py - Audit & Statistic Verification for Phase 6J-A Features

Audits the 4 proposed 6J-A features across train.csv (886) and val.csv (190):
  1. causal_relation_density: causality_count / sentence_count
  2. example_density: example_count / sentence_count
  3. claim_support_ratio: sentences with claim+support markers / sentence_count
  4. task2_length_shortfall: max(0.0, 250.0 - word_count)

Checks:
  - min, max, mean, std, zero_rate
  - Pearson correlation with true band
  - Mean & Median values across Low (4.0-5.5), Mid (6.0-7.0), High (7.5-9.0)

STRICT RULE: test.csv is LOCKED.
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from evaluate import _load_dataset

# Extended causality & example phrases
_CAUSALITY_PHRASES_EXT = [
    "because", "therefore", "as a result", "consequently",
    "this means", "which means", "thus", "hence", "due to", "owing to",
    "since", "so that", "in order to", "leads to", "results in",
    "this stems from", "this brings about", "as a consequence",
    "the reason is", "the primary cause", "this accounts for", "giving rise to", "thereby"
]

_EXAMPLE_PHRASES_EXT = [
    "for example", "for instance", "such as", "to illustrate",
    "a case in point", "one clear example", "a prominent example",
    "as an illustration", "take for example", "take for instance",
    "studies show", "research suggests", "statistics indicate", "according to",
    "evidence suggests", "research shows"
]

_CAUSALITY_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in sorted(_CAUSALITY_PHRASES_EXT, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_EXAMPLE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in sorted(_EXAMPLE_PHRASES_EXT, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\b[a-z']+\b")

def compute_6j_a_features(essay: str, prompt: str) -> dict:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(essay) if s.strip()]
    sent_count = max(len(sentences), 1)
    word_count = len(_WORD_RE.findall(essay.lower()))

    causality_count = len(_CAUSALITY_RE.findall(essay))
    example_count = len(_EXAMPLE_RE.findall(essay))

    causal_relation_density = round(causality_count / sent_count, 4)
    example_density = round(example_count / sent_count, 4)

    # Claim support ratio: sentences that contain a cause OR an example marker
    supported_sents = 0
    for s in sentences:
        if _CAUSALITY_RE.search(s) or _EXAMPLE_RE.search(s):
            supported_sents += 1
    claim_support_ratio = round(supported_sents / sent_count, 4)

    task2_length_shortfall = max(0.0, float(250 - word_count))

    return {
        "causal_relation_density": causal_relation_density,
        "example_density": example_density,
        "claim_support_ratio": claim_support_ratio,
        "task2_length_shortfall": task2_length_shortfall,
    }

def main():
    print("=" * 80)
    print("  PHASE 6J-A FEATURE STATISTICAL AUDIT")
    print("=" * 80)

    train_data = _load_dataset(Path("data/train.csv"))
    val_data = _load_dataset(Path("data/val.csv"))
    combined = train_data + val_data

    records = []
    for sample in tqdm(combined, desc="Auditing 6J-A features"):
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        band = float(sample.get("band", 5.0))
        if not essay.strip() or not prompt.strip():
            continue
        feats = compute_6j_a_features(essay, prompt)
        feats["band"] = band
        records.append(feats)

    df = pd.DataFrame(records)
    print(f"\nExtracted 6J-A candidate features for {len(df)} samples (Train + Val).")

    feat_names = ["causal_relation_density", "example_density", "claim_support_ratio", "task2_length_shortfall"]

    print("\n  1. Overall Distribution & Pearson Correlation with Band:")
    print("  " + "-" * 75)
    print(f"  {'Feature':<25} | {'Min':>7} | {'Max':>7} | {'Mean':>7} | {'Std':>7} | {'Zero%':>6} | {'Pearson r':>9}")
    print("  " + "-" * 75)
    for feat in feat_names:
        s = df[feat]
        min_v, max_v = s.min(), s.max()
        mean_v, std_v = s.mean(), s.std()
        zero_pct = (s == 0).mean() * 100
        r_val = s.corr(df["band"])
        print(f"  {feat:<25} | {min_v:>7.2f} | {max_v:>7.2f} | {mean_v:>7.4f} | {std_v:>7.4f} | {zero_pct:>5.1f}% | {r_val:>+9.4f}")
    print("  " + "-" * 75)

    print("\n  2. Mean & Median Values across Band Groups:")
    print("  " + "-" * 75)
    groups = {
        "Low (4.0-5.5)": df[df["band"] <= 5.5],
        "Mid (6.0-7.0)": df[(df["band"] >= 6.0) & (df["band"] <= 7.0)],
        "High (7.5-9.0)": df[df["band"] >= 7.5],
    }

    for feat in feat_names:
        print(f"  Feature: {feat}")
        for g_name, g_df in groups.items():
            s = g_df[feat]
            print(f"    {g_name:<15}: Mean = {s.mean():>7.4f} | Median = {s.median():>7.4f} | 75% = {s.quantile(0.75):>7.4f}")
        print()
    print("=" * 80)

if __name__ == "__main__":
    main()
