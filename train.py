"""
train.py — Full Training Pipeline
Loads dataset → extracts features → trains ML model → calibrates → evaluates.

Usage:
    python train.py
    python train.py --dataset data/dataset.json
"""

import csv
import json
import sys
import argparse
import numpy as np
from pathlib import Path

# ── Scorer modules ─────────────────────────────────────────
from scorer.features_text import extract_text_features
from scorer.features_essay import extract_essay_features
from scorer.features_task import extract_task_features
from scorer.features_antigaming import extract_antigaming_features
from scorer.normalizer import normalize_features, compute_stats_from_dataset
from scorer.ml_model import IELTSModel
from scorer.calibration import train_calibrator


# ── Feature Keys (sorted alphabetically for consistency) ──

FEATURE_KEYS = sorted([
    "argument_depth_score",
    "awl_ratio",
    "complex_clause_ratio",
    "connectors_per_sentence",
    "discourse_marker_count",
    "discourse_marker_variety",
    "gaming_penalty",
    "grammar_error_rate",
    "has_conclusion",
    "has_intro",
    "mattr",
    "max_word_repetition_ratio",
    "paragraph_count",
    "rare_word_ratio",
    "semantic_coherence",
    "sent_length_variance",
    "sentence_count",
    "task_coverage_score",
    "tense_consistent",
    "topic_similarity",
    "word_count",
])


def extract_all_features(essay: str, prompt: str) -> dict:
    """
    Run all 4 feature extractors and merge into a flat dict.
    Boolean values are converted to 0/1 for ML.
    """
    text_feat = extract_text_features(essay)
    essay_feat = extract_essay_features(essay, prompt)
    task_feat = extract_task_features(essay, prompt)
    antigaming_feat = extract_antigaming_features(essay, prompt)

    # Merge into flat dict
    merged = {}

    # From features_text
    merged["mattr"] = text_feat.get("mattr", 0.0)
    merged["rare_word_ratio"] = text_feat.get("rare_word_ratio", 0.0)
    merged["awl_ratio"] = text_feat.get("awl_ratio", 0.0)
    merged["grammar_error_rate"] = text_feat.get("grammar_error_rate", 0.0)
    merged["complex_clause_ratio"] = text_feat.get("complex_clause_ratio", 0.0)
    merged["sent_length_variance"] = text_feat.get("sent_length_variance", 0.0)
    merged["tense_consistent"] = 1.0 if text_feat.get("tense_consistent", True) else 0.0
    merged["max_word_repetition_ratio"] = text_feat.get("max_word_repetition_ratio", 0.0)

    # From features_essay
    merged["paragraph_count"] = float(essay_feat.get("paragraph_count", 1))
    merged["word_count"] = float(essay_feat.get("word_count", 0))
    merged["sentence_count"] = float(essay_feat.get("sentence_count", 0))
    merged["has_intro"] = 1.0 if essay_feat.get("has_intro", False) else 0.0
    merged["has_conclusion"] = 1.0 if essay_feat.get("has_conclusion", False) else 0.0
    merged["semantic_coherence"] = essay_feat.get("semantic_coherence", 0.0)
    merged["topic_similarity"] = essay_feat.get("topic_similarity", 0.0)
    merged["discourse_marker_count"] = float(essay_feat.get("discourse_marker_count", 0))
    merged["discourse_marker_variety"] = essay_feat.get("discourse_marker_variety", 0.0)

    # From features_task
    merged["task_coverage_score"] = task_feat.get("task_coverage_score", 0.5)
    merged["argument_depth_score"] = task_feat.get("argument_depth_score", 0.0)

    # From features_antigaming
    merged["connectors_per_sentence"] = antigaming_feat.get("connectors_per_sentence", 0.0)
    merged["gaming_penalty"] = antigaming_feat.get("gaming_penalty", 0.0)

    return merged


def features_to_array(features: dict, feature_keys: list) -> np.ndarray:
    """Convert feature dict to numpy array in consistent order."""
    return np.array([features.get(k, 0.0) for k in feature_keys], dtype=np.float64)


def _load_dataset(path: Path) -> list:
    """Load dataset from CSV or JSON, normalizing column names.

    CSV column mapping (from user's dataset):
        Question       → prompt
        Essay          → essay
        Overall        → band
        Task_Response  → task_response
        Coherence_Cohesion → coherence
        Lexical_Resource   → lexical
        Range_Accuracy     → grammar
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            dataset = []
            for row in reader:
                overall = row.get("Overall", "").strip()
                if not overall:
                    continue
                try:
                    band = float(overall)
                except ValueError:
                    continue

                sample = {
                    "essay": row.get("Essay", "").strip(),
                    "prompt": row.get("Question", "").strip(),
                    "band": band,
                }
                # Per-criterion scores (if available)
                for csv_col, key in [
                    ("Task_Response", "task_response"),
                    ("Coherence_Cohesion", "coherence"),
                    ("Lexical_Resource", "lexical"),
                    ("Range_Accuracy", "grammar"),
                ]:
                    val = row.get(csv_col, "").strip()
                    if val:
                        try:
                            sample[key] = float(val)
                        except ValueError:
                            pass
                dataset.append(sample)
        return dataset

    else:
        # JSON format
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Train IELTS Writing Scorer v2")
    parser.add_argument("--dataset", default="data/train.csv", help="Path to training dataset (CSV)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        print("Please run split_dataset.py first to generate data/train.csv")
        sys.exit(1)

    # ── Step 1: Load dataset ──────────────────────────────
    print("=" * 60)
    print("  IELTS Writing Scorer v2 — Training Pipeline")
    print("=" * 60)
    print(f"\n[1/7] Loading dataset: {dataset_path}")

    dataset = _load_dataset(dataset_path)
    print(f"      Loaded {len(dataset)} essays")

    # ── Step 2: Extract features ──────────────────────────
    print(f"\n[2/7] Extracting features from {len(dataset)} essays...")

    all_features = []
    skipped = 0

    for i, sample in enumerate(dataset):
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")

        if not essay.strip() or not prompt.strip():
            print(f"      [SKIP] Sample {i}: empty essay or prompt")
            skipped += 1
            continue

        try:
            feats = extract_all_features(essay, prompt)
            all_features.append(feats)
            print(f"      [{i+1}/{len(dataset)}] OK — {feats.get('word_count', 0):.0f} words")
        except Exception as e:
            print(f"      [{i+1}/{len(dataset)}] ERROR: {e}")
            skipped += 1

    if not all_features:
        print("[ERROR] No features extracted. Check dataset format.")
        sys.exit(1)

    print(f"      Extracted: {len(all_features)}, Skipped: {skipped}")

    # ── Step 3: Compute feature stats ─────────────────────
    print("\n[3/7] Computing feature statistics...")

    stats = compute_stats_from_dataset(all_features)

    stats_path = Path("data/feature_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"      Saved: {stats_path}")

    # ── Step 4: Normalize features ────────────────────────
    print("\n[4/7] Normalizing features...")

    normalized_list = []
    for feats in all_features:
        norm = normalize_features(feats)
        normalized_list.append(norm)

    # Build feature matrix
    X = np.array([
        features_to_array(norm, FEATURE_KEYS)
        for norm in normalized_list
    ])

    print(f"      Feature matrix shape: {X.shape}")

    # ── Step 5: Prepare targets ───────────────────────────
    print("\n[5/7] Preparing ML targets...")

    # Filter dataset to match extracted features (skip same indices)
    valid_samples = [s for s in dataset if s.get("essay", "").strip() and s.get("prompt", "").strip()]

    # ── Task 6: Detect sub-score label availability ──
    # Count how many samples have REAL sub-score labels (not NaN/missing)
    sub_score_keys = {
        "task_response": "task_response",
        "lexical": "lexical",
        "grammar": "grammar",
        "coherence": "coherence",
    }
    sub_score_counts = {k: 0 for k in sub_score_keys}
    for sample in valid_samples[:len(all_features)]:
        for key in sub_score_keys:
            if key in sample and sample[key] is not None and not (isinstance(sample[key], float) and np.isnan(sample[key])):
                sub_score_counts[key] += 1

    total = len(valid_samples[:len(all_features)])
    has_real_subscores = all(c > total * 0.5 for c in sub_score_counts.values())

    y = {}
    bands = [sample.get("band", 5.0) for sample in valid_samples[:len(all_features)]]
    y["overall"] = np.array(bands, dtype=np.float64)

    if has_real_subscores:
        print("      [MODE] INDEPENDENT sub-score training + Overall model (Mode B)")
        print(f"      Sub-score label coverage: {sub_score_counts}")
        y["task_response"] = np.array([sample.get("task_response", b) for sample, b in zip(valid_samples[:len(all_features)], bands)], dtype=np.float64)
        y["lexical"] = np.array([sample.get("lexical", b) for sample, b in zip(valid_samples[:len(all_features)], bands)], dtype=np.float64)
        y["grammar"] = np.array([sample.get("grammar", b) for sample, b in zip(valid_samples[:len(all_features)], bands)], dtype=np.float64)
        y["coherence"] = np.array([sample.get("coherence", b) for sample, b in zip(valid_samples[:len(all_features)], bands)], dtype=np.float64)
    else:
        print("      [MODE] OVERALL-ONLY training (Mode A, sub-score labels unavailable)")
        print(f"      Sub-score label coverage: {sub_score_counts}")
        print("      Only 1 model will be trained on the Overall band.")
        print("      Sub-scores will be estimated dynamically using feature analysis.")

    for key in y:
        y[key] = np.array(y[key], dtype=np.float64)
        print(f"      {key}: mean={y[key].mean():.2f}, std={y[key].std():.2f}, "
              f"range=[{y[key].min():.1f}, {y[key].max():.1f}]")

    # ── Step 6: Train ML model ────────────────────────────
    print("\n[6/7] Training LightGBM models...")

    model = IELTSModel()
    model.train(X, y, FEATURE_KEYS)

    model_path = "data/ml_model.pkl"
    model.save(model_path)
    print(f"      Saved: {model_path}")

    # ── Step 7: Train calibrator ──────────────────────────
    print("\n[7/7] Training isotonic calibrator...")

    # Predict on training data
    predictions = model.predict(X)

    # Compute raw overall scores after constraints for calibration input
    from scorer.scorer import score_essay
    from scorer.constraint import apply_constraints

    raw_overalls = []
    true_bands = []
    for i, sample in enumerate(valid_samples[:len(all_features)]):
        norm_feat = normalized_list[i]
        raw_feat = all_features[i]
        ml_preds = {k: predictions[k][i] for k in predictions}

        # Run gating & constraints pipeline to get constrained overall
        score_res = score_essay(norm_feat, raw_feat, ml_preds)
        gated_scores = score_res["scores"]
        constrained_scores, _ = apply_constraints(gated_scores, raw_feat)
        
        raw_overall = constrained_scores["overall"]
        raw_overalls.append(float(raw_overall))
        true_bands.append(float(sample.get("band", 5.0)))

    cal_path = "data/calibration_model.pkl"
    train_calibrator(raw_overalls, true_bands, save_path=cal_path)
    print(f"      Saved: {cal_path}")

    # ── Quick evaluation on training data ─────────────────
    print("\n" + "=" * 60)
    print("  TRAINING EVALUATION (on training data)")
    print("=" * 60)

    from scorer.calibration import calibrate

    errors = []
    within_half = 0

    print(f"\n  {'#':>3}  {'True':>5}  {'Pred':>5}  {'Err':>5}  {'±0.5':>5}")
    print("  " + "-" * 30)

    for i in range(len(raw_overalls)):
        pred_band = calibrate(raw_overalls[i])
        true_band = true_bands[i]
        error = abs(pred_band - true_band)
        errors.append(error)

        if error <= 0.5:
            within_half += 1

        marker = "OK" if error <= 0.5 else "ERR"
        print(f"  {i+1:>3}  {true_band:>5.1f}  {pred_band:>5.1f}  {error:>5.1f}  {marker:>5}")

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    accuracy = within_half / len(errors) * 100 if errors else 0

    print(f"\n  MAE:           {mae:.3f}")
    print(f"  RMSE:          {rmse:.3f}")
    print(f"  ±0.5 Accuracy: {accuracy:.1f}%")
    print(f"  Samples:       {len(errors)}")
    print("=" * 60)

    if mae < 0.5:
        print("\n  [PASS] MAE < 0.5 - TARGET MET")
    else:
        print("\n  [WARN] MAE >= 0.5 - needs more data or tuning")

    if accuracy >= 75:
        print("  [PASS] ±0.5 Accuracy >= 75% - TARGET MET")
    else:
        print("  [WARN] ±0.5 Accuracy < 75% - needs more data or tuning")

    # Save model info metadata
    import datetime
    model_info = {
        "algorithm": "LightGBM",
        "dataset_size": len(all_features),
        "feature_count": len(FEATURE_KEYS),
        "mae": float(mae),
        "accuracy_05": float(accuracy),
        "feature_version": IELTSModel.FEATURE_VERSION,
        "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    info_path = Path("data/model_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    print(f"\n  Training complete! Models saved in data/")
    print(f"  Metadata saved to: {info_path}")
    print(f"  Run 'python app.py' to start the API.")


if __name__ == "__main__":
    main()
