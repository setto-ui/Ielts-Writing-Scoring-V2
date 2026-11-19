"""
retrain_baselines_6j.py - Retrain Model A (21 feat) and Model F (27 feat) on updated pipeline.
Needed because features_task.py was updated (new argument_depth_score formula, new features).
All three models (A, F, 6JA) must use the SAME feature_stats.json for fair comparison.
STRICT RULE: test.csv is LOCKED.
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from evaluate import _load_dataset, extract_all_features, features_to_array
from scorer.normalizer import normalize_features
from scorer.ml_model import IELTSModel

# ---- Model A: original 21 features ----
FEATURE_KEYS_A = sorted([
    "argument_depth_score", "awl_ratio", "complex_clause_ratio",
    "connectors_per_sentence", "discourse_marker_count",
    "discourse_marker_variety", "gaming_penalty", "grammar_error_rate",
    "has_conclusion", "has_intro", "mattr", "max_word_repetition_ratio",
    "paragraph_count", "rare_word_ratio", "semantic_coherence",
    "sent_length_variance", "sentence_count", "task_coverage_score",
    "tense_consistent", "topic_similarity", "word_count",
])

# ---- Model F: 27 features (Phase 6E set) ----
FEATURE_KEYS_F = sorted([
    "argument_depth_score", "avg_paragraph_length", "avg_sentence_length",
    "awl_ratio", "complex_clause_ratio", "connectors_per_sentence",
    "discourse_marker_count", "discourse_marker_variety", "gaming_penalty",
    "grammar_error_density", "grammar_error_rate", "has_conclusion",
    "has_intro", "long_word_ratio", "mattr", "max_word_repetition_ratio",
    "paragraph_count", "prompt_keyword_overlap", "rare_word_ratio",
    "semantic_coherence", "sent_length_variance", "sentence_count",
    "sentence_length_std", "task_coverage_score", "tense_consistent",
    "topic_similarity", "word_count",
])

def train_and_save(feature_keys, model_name, output_path, train_data):
    print(f"\n--- Training {model_name} ({len(feature_keys)} features) ---")
    features_list = []
    y_list = []

    for sample in tqdm(train_data):
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        band = float(sample.get("band", 5.0))
        if not essay.strip() or not prompt.strip():
            continue
        raw_feat = extract_all_features(essay, prompt)
        norm_feat = normalize_features(raw_feat)
        arr = features_to_array(norm_feat, feature_keys)
        features_list.append(arr)
        y_list.append(band)

    X = np.vstack(features_list)
    y_dict = {"overall": np.array(y_list)}
    print(f"  Shape: {X.shape}")

    model = IELTSModel()
    model.train(X=X, y=y_dict, feature_names=feature_keys, objective="regression", sample_weight=None)
    model.save(output_path)
    print(f"  Saved to {output_path}")

def main():
    print("=" * 70)
    print("  RETRAIN BASELINES — Updated Feature Pipeline (6J-A)")
    print("=" * 70)

    train_data = _load_dataset(Path("data/train.csv"))
    print(f"Loaded {len(train_data)} training samples.")

    train_and_save(FEATURE_KEYS_A, "Model A", "data/model_exp_A_6j.pkl", train_data)
    train_and_save(FEATURE_KEYS_F, "Model F", "data/model_exp_F_6j.pkl", train_data)

    print("\nDone. Retrained baselines saved as model_exp_A_6j.pkl and model_exp_F_6j.pkl.")
    print("Use these for fair comparison against model_exp_6JA.pkl.")

if __name__ == "__main__":
    main()
