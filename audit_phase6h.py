"""
audit_phase6h.py - Phase 6H: Feature Quality & Task-Type Audit

Strictly reads data/train.csv (886) and data/val.csv (190).
test.csv is LOCKED and NOT touched.

Audits:
  H1: Task 1 vs Task 2 Detection & Dataset Audit
  H2: semantic_coherence Distribution & Quality Audit
  H3: argument_depth_score Deep Dive & Distribution Audit
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from evaluate import _load_dataset, extract_all_features
from scorer.normalizer import normalize_features

# Task 1 detection rules (inspecting prompt text & essay characteristics)
TASK1_PROMPT_KEYWORDS = [
    "summarise the information", "selecting and reporting", "make comparisons",
    "the diagram", "the chart", "the table", "the graph", "the map",
    "the process", "the illustration", "the plan", "student accommodation",
    "bar chart", "line graph", "pie chart", "flow chart"
]

def is_task1(prompt: str, essay: str) -> bool:
    prompt_lower = prompt.lower()
    for kw in TASK1_PROMPT_KEYWORDS:
        if kw in prompt_lower:
            return True
    return False

def main():
    print("=" * 80)
    print("  PHASE 6H: FEATURE QUALITY & TASK-TYPE AUDIT")
    print("=" * 80)

    # Load datasets (train & val only)
    train_data = _load_dataset(Path("data/train.csv"))
    val_data = _load_dataset(Path("data/val.csv"))

    def process_samples(dataset, set_name):
        records = []
        for idx, s in enumerate(tqdm(dataset, desc=f"Extracting {set_name}")):
            essay = s.get("essay", "")
            prompt = s.get("prompt", "")
            band = float(s.get("band", 5.0))
            if not essay.strip() or not prompt.strip():
                continue

            raw_feat = extract_all_features(essay, prompt)
            task1_flag = is_task1(prompt, essay)

            row = {
                "set": set_name,
                "id": idx,
                "band": band,
                "prompt": prompt,
                "essay": essay,
                "word_count": raw_feat.get("word_count", 0),
                "is_task1": task1_flag,
                "semantic_coherence": raw_feat.get("semantic_coherence", 0.0),
                "topic_similarity": raw_feat.get("topic_similarity", 0.0),
                "argument_depth_score": raw_feat.get("argument_depth_score", 0.0),
                "causality_count": raw_feat.get("causality_count", 0.0),
                "example_count": raw_feat.get("example_count", 0.0),
                "explanation_chain_ratio": raw_feat.get("explanation_chain_ratio", 0.0),
                "long_word_ratio": raw_feat.get("long_word_ratio", 0.0),
                "grammar_error_rate": raw_feat.get("grammar_error_rate", 0.0),
            }
            records.append(row)
        return pd.DataFrame(records)

    df_train = process_samples(train_data, "Train")
    df_val = process_samples(val_data, "Val")
    df_combined = pd.concat([df_train, df_val], ignore_index=True)

    print("\n" + "=" * 80)
    print("  H1: TASK 1 vs TASK 2 DATASET AUDIT")
    print("=" * 80)
    
    for s_name, df_set in [("Train (886)", df_train), ("Val (190)", df_val), ("Combined (1076)", df_combined)]:
        t1_count = df_set["is_task1"].sum()
        t2_count = len(df_set) - t1_count
        t1_pct = (t1_count / len(df_set)) * 100
        print(f"  {s_name:<16}: Total={len(df_set):>4} | Task 1={t1_count:>3} ({t1_pct:>5.1f}%) | Task 2={t2_count:>4}")

    print("\n  Task 1 Distribution by Band (Combined Train + Val):")
    band_groups = [
        ("Low (4.0-5.5)", df_combined[df_combined["band"] <= 5.5]),
        ("Mid (6.0-7.0)", df_combined[(df_combined["band"] >= 6.0) & (df_combined["band"] <= 7.0)]),
        ("High (7.5-9.0)", df_combined[df_combined["band"] >= 7.5]),
    ]
    for b_name, b_df in band_groups:
        t1_c = b_df["is_task1"].sum()
        t1_p = (t1_c / len(b_df)) * 100 if len(b_df) > 0 else 0
        mean_words_t1 = b_df[b_df["is_task1"]]["word_count"].mean() if t1_c > 0 else 0
        mean_words_t2 = b_df[~b_df["is_task1"]]["word_count"].mean() if (len(b_df) - t1_c) > 0 else 0
        print(f"    {b_name:<15}: Total={len(b_df):>3} | Task 1={t1_c:>2} ({t1_p:>4.1f}%) [Avg words T1={mean_words_t1:.0f}, T2={mean_words_t2:.0f}]")

    print("\n" + "=" * 80)
    print("  H2: SEMANTIC_COHERENCE AUDIT")
    print("=" * 80)
    
    sc = df_combined["semantic_coherence"]
    corr_sc = df_combined["semantic_coherence"].corr(df_combined["band"])
    print(f"  Overall Distribution (N={len(df_combined)}):")
    print(f"    Min={sc.min():.4f}, Max={sc.max():.4f}, Mean={sc.mean():.4f}, Std={sc.std():.4f}")
    print(f"    25%={sc.quantile(0.25):.4f}, Median={sc.quantile(0.50):.4f}, 75%={sc.quantile(0.75):.4f}")
    print(f"    Pearson Correlation with True Band: {corr_sc:+.4f}")
    
    print("\n  semantic_coherence Percentiles by Band Group:")
    for b_name, b_df in band_groups:
        bsc = b_df["semantic_coherence"]
        print(f"    {b_name:<15}: Mean={bsc.mean():.4f} | Std={bsc.std():.4f} | 25%={bsc.quantile(0.25):.4f} | Med={bsc.median():.4f} | 75%={bsc.quantile(0.75):.4f}")

    print("\n" + "=" * 80)
    print("  H3: ARGUMENT_DEPTH_SCORE AUDIT")
    print("=" * 80)
    
    ad = df_combined["argument_depth_score"]
    corr_ad = df_combined["argument_depth_score"].corr(df_combined["band"])
    print(f"  Overall Distribution (N={len(df_combined)}):")
    print(f"    Min={ad.min():.4f}, Max={ad.max():.4f}, Mean={ad.mean():.4f}, Std={ad.std():.4f}")
    print(f"    25%={ad.quantile(0.25):.4f}, Median={ad.quantile(0.50):.4f}, 75%={ad.quantile(0.75):.4f}")
    print(f"    Pearson Correlation with True Band: {corr_ad:+.4f}")

    print("\n  argument_depth_score Percentiles & Sub-components by Band Group:")
    for b_name, b_df in band_groups:
        bad = b_df["argument_depth_score"]
        c_cnt = b_df["causality_count"].mean()
        e_cnt = b_df["example_count"].mean()
        chain = b_df["explanation_chain_ratio"].mean()
        print(f"    {b_name:<15}: Mean={bad.mean():.4f} | Med={bad.median():.4f} | 75%={bad.quantile(0.75):.4f} | Causality={c_cnt:.2f} | Examples={e_cnt:.2f} | Chain={chain:.3f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
