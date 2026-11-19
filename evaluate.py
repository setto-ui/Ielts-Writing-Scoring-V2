"""
evaluate.py — Evaluation Script
Runs full pipeline on dataset and computes metrics: MAE, RMSE, ±0.5 accuracy.

Usage:
    python evaluate.py
    python evaluate.py --dataset data/dataset.json
"""

import csv
import json
import sys
import argparse
import numpy as np
from pathlib import Path

from scorer.features_text import extract_text_features
from scorer.features_essay import extract_essay_features
from scorer.features_task import extract_task_features
from scorer.features_antigaming import extract_antigaming_features
from scorer.features_lexical import extract_lexical_features
from scorer.normalizer import normalize_features
from scorer.ml_model import IELTSModel
from scorer.scorer import score_essay, round_to_ielts_band
from scorer.constraint import apply_constraints
from scorer.calibration import calibrate


# Feature keys (must match train.py)
# 31-feature set — Model 6J-B baseline (Phase 6J champion feature set)
# NOTE: 6J-C features (connective_density, paragraph_structure_balance, lexical_overlap)
# were audited and found to HURT validation (Overall MAE +0.013, Overall +/-0.5 dropped 63.7% -> 59.5%).
FEATURE_KEYS = sorted([
    "academic_word_ratio",
    "argument_depth_score",
    "avg_paragraph_length",
    "avg_sentence_length",
    "awl_ratio",
    "complex_clause_ratio",
    "connectors_per_sentence",
    "discourse_marker_count",
    "discourse_marker_variety",
    "gaming_penalty",
    "grammar_error_density",
    "grammar_error_rate",
    "has_conclusion",
    "has_intro",
    "lexical_density",
    "lexical_diversity",
    "long_word_ratio",
    "mattr",
    "max_word_repetition_ratio",
    "paragraph_count",
    "prompt_keyword_overlap",
    "rare_word_ratio",
    "repetition_rate",
    "semantic_coherence",
    "sent_length_variance",
    "sentence_count",
    "sentence_length_std",
    "task_coverage_score",
    "tense_consistent",
    "topic_similarity",
    "word_count",
])


def extract_all_features(essay: str, prompt: str) -> dict:
    """Run all feature extractors and merge into flat dict."""
    text_feat = extract_text_features(essay)
    essay_feat = extract_essay_features(essay, prompt)
    task_feat = extract_task_features(essay, prompt)
    antigaming_feat = extract_antigaming_features(essay, prompt)
    lexical_feat = extract_lexical_features(essay)

    merged = {}
    merged["mattr"] = text_feat.get("mattr", 0.0)
    merged["rare_word_ratio"] = text_feat.get("rare_word_ratio", 0.0)
    merged["awl_ratio"] = text_feat.get("awl_ratio", 0.0)
    merged["long_word_ratio"] = text_feat.get("long_word_ratio", 0.0)
    merged["grammar_error_rate"] = text_feat.get("grammar_error_rate", 0.0)
    merged["grammar_error_density"] = text_feat.get("grammar_error_density", 0.0)
    merged["complex_clause_ratio"] = text_feat.get("complex_clause_ratio", 0.0)
    merged["sent_length_variance"] = text_feat.get("sent_length_variance", 0.0)
    merged["avg_sentence_length"] = text_feat.get("avg_sentence_length", 0.0)
    merged["sentence_length_std"] = text_feat.get("sentence_length_std", 0.0)
    merged["tense_consistent"] = 1.0 if text_feat.get("tense_consistent", True) else 0.0
    merged["max_word_repetition_ratio"] = text_feat.get("max_word_repetition_ratio", 0.0)
    merged["paragraph_count"] = float(essay_feat.get("paragraph_count", 1))
    merged["avg_paragraph_length"] = essay_feat.get("avg_paragraph_length", 0.0)
    merged["word_count"] = float(essay_feat.get("word_count", 0))
    merged["sentence_count"] = float(essay_feat.get("sentence_count", 0))
    merged["has_intro"] = 1.0 if essay_feat.get("has_intro", False) else 0.0
    merged["has_conclusion"] = 1.0 if essay_feat.get("has_conclusion", False) else 0.0
    merged["semantic_coherence"] = essay_feat.get("semantic_coherence", 0.0)
    merged["topic_similarity"] = essay_feat.get("topic_similarity", 0.0)
    merged["discourse_marker_count"] = float(essay_feat.get("discourse_marker_count", 0))
    merged["discourse_marker_variety"] = essay_feat.get("discourse_marker_variety", 0.0)
    merged["task_coverage_score"] = task_feat.get("task_coverage_score", 0.5)
    merged["argument_depth_score"] = task_feat.get("argument_depth_score", 0.0)
    merged["prompt_keyword_overlap"] = task_feat.get("prompt_keyword_overlap", 0.0)
    merged["connectors_per_sentence"] = antigaming_feat.get("connectors_per_sentence", 0.0)
    merged["gaming_penalty"] = antigaming_feat.get("gaming_penalty", 0.0)
    # Phase 6J-B Lexical features
    merged["lexical_diversity"] = lexical_feat.get("lexical_diversity", 0.0)
    merged["lexical_density"] = lexical_feat.get("lexical_density", 0.0)
    merged["repetition_rate"] = lexical_feat.get("repetition_rate", 0.0)
    merged["academic_word_ratio"] = lexical_feat.get("academic_word_ratio", 0.0)

    return merged


def features_to_array(features: dict, feature_keys: list) -> np.ndarray:
    """Convert feature dict to numpy array in consistent order."""
    return np.array([features.get(k, 0.0) for k in feature_keys], dtype=np.float64)


def run_full_pipeline(essay: str, prompt: str, model: IELTSModel) -> dict:
    """Run the complete scoring pipeline on a single essay."""
    # 1. Extract features
    raw_features = extract_all_features(essay, prompt)

    # 2. Normalize
    norm_features = normalize_features(raw_features)

    # 3. ML predict
    X = features_to_array(norm_features, FEATURE_KEYS).reshape(1, -1)
    ml_preds = model.predict(X)
    # ml_preds is {criterion: float} for single sample

    # 4. Score (hard gates)
    # score_essay returns {"scores": {...}, "bands": {...}, "gates": [...]}
    score_result = score_essay(
        normalized_features=norm_features,
        raw_features=raw_features,
        ml_predictions=ml_preds,
    )
    gated_scores = score_result["scores"]

    # 5. Constraints
    constrained_scores, rules = apply_constraints(gated_scores, raw_features)

    # 6. Calibrate
    raw_overall = constrained_scores.get("overall")
    overall_band = calibrate(raw_overall)

    return {
        "overall_band": overall_band,
        "sub_scores": constrained_scores,
        "raw_features": raw_features,
        "applied_constraints": rules,
    }


def _load_dataset(path: Path) -> list:
    """Load dataset from CSV or JSON, normalizing column names."""
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
                # Per-criterion sub-scores
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
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate IELTS Writing Scorer v2")
    parser.add_argument("--dataset", default="data/test.csv", help="Path to evaluation dataset (CSV)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        sys.exit(1)

    # Load model
    model_path = "data/ml_model.pkl"
    try:
        model = IELTSModel.load(model_path)
    except FileNotFoundError:
        print(f"[ERROR] Model not found: {model_path}")
        print("Run 'python train.py' first.")
        sys.exit(1)

    # Load dataset
    dataset = _load_dataset(dataset_path)

    print("=" * 70)
    print("  IELTS Writing Scorer v2 — Evaluation")
    print("=" * 70)
    print(f"  Dataset: {dataset_path} ({len(dataset)} essays)")
    print()

    # Evaluate
    errors = []
    sub_errors = {"task_response": [], "lexical": [], "grammar": [], "coherence": []}
    results = []

    for i, sample in enumerate(dataset):
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        true_band = sample.get("band", 5.0)

        if not essay.strip() or not prompt.strip():
            continue

        try:
            result = run_full_pipeline(essay, prompt, model)
            pred_band = result["overall_band"]
            error = abs(pred_band - true_band)
            errors.append(error)

            # Sub-score errors (if available)
            for key in sub_errors:
                true_sub = sample.get(key)
                if true_sub is not None:
                    pred_sub = result["sub_scores"].get(key, 0)
                    sub_errors[key].append(abs(pred_sub - true_sub))

            results.append({
                "index": i + 1,
                "true": true_band,
                "pred": pred_band,
                "error": error,
                "sub_scores": result["sub_scores"],
                "constraints": result["applied_constraints"],
            })

        except Exception as e:
            print(f"  [ERROR] Sample {i+1}: {e}")

    if not errors:
        print("  No valid results to evaluate.")
        sys.exit(1)

    # Print results table
    print(f"  {'#':>3}  {'True':>5}  {'Pred':>5}  {'Err':>5}  {'TR':>4}  {'LR':>4}  {'GR':>4}  {'CC':>4}  {'Rules':>6}")
    print("  " + "-" * 55)

    for r in results:
        s = r["sub_scores"]
        marker = "OK" if r["error"] <= 0.5 else "ERR"
        rules_count = len(r["constraints"])
        print(
            f"  {r['index']:>3}  {r['true']:>5.1f}  {r['pred']:>5.1f}  "
            f"{r['error']:>5.1f} [{marker}]  "
            f"{s.get('task_response', 0):>4.1f}  {s.get('lexical', 0):>4.1f}  "
            f"{s.get('grammar', 0):>4.1f}  {s.get('coherence', 0):>4.1f}  "
            f"{rules_count:>6}"
        )

    # Summary metrics
    errors_np = np.array(errors)
    mae = float(np.mean(errors_np))
    rmse = float(np.sqrt(np.mean(errors_np ** 2)))
    within_half = sum(1 for e in errors if e <= 0.5)
    accuracy = within_half / len(errors) * 100

    print("\n" + "=" * 70)
    print("  OVERALL METRICS")
    print("=" * 70)
    print(f"  MAE:              {mae:.3f}")
    print(f"  RMSE:             {rmse:.3f}")
    print(f"  ±0.5 Accuracy:    {accuracy:.1f}% ({within_half}/{len(errors)})")
    print(f"  Max Error:        {max(errors):.2f}")
    print(f"  Samples:          {len(errors)}")

    # Per-band metrics
    print("\n" + "=" * 70)
    print("  PER-BAND METRICS")
    print("=" * 70)
    print(f"  {'Band':>4} | {'N':>4} | {'MAE':>5} | {'Bias':>6} | {'±0.5 Acc':>8}")
    print("  " + "-" * 43)
    
    bands_dict = {}
    for r in results:
        b = r["true"]
        if b not in bands_dict:
            bands_dict[b] = []
        bands_dict[b].append((r["pred"], r["error"]))
        
    for b in sorted(bands_dict.keys()):
        preds_errors = bands_dict[b]
        n = len(preds_errors)
        b_mae = np.mean([e for p, e in preds_errors])
        b_bias = np.mean([p - b for p, e in preds_errors])
        b_acc = sum(1 for p, e in preds_errors if e <= 0.5) / n * 100
        print(f"  {b:>4.1f} | {n:>4} | {b_mae:>5.3f} | {b_bias:>6.3f} | {b_acc:>7.1f}%")

    # Sub-score MAE
    print("\n  Sub-score MAE:")
    for key in sub_errors:
        if sub_errors[key]:
            sub_mae = np.mean(sub_errors[key])
            print(f"    {key:>20}: {sub_mae:.3f}")

    print("=" * 70)

    # Pass/fail
    print()
    if mae < 0.5:
        print("  [PASS] MAE < 0.5 — TARGET MET")
    else:
        print("  [WARN] MAE >= 0.5 — needs more data or tuning")
    if accuracy >= 75:
        print("  [PASS] ±0.5 Accuracy >= 75% — TARGET MET")
    else:
        print("  [WARN] ±0.5 Accuracy < 75% — needs more data or tuning")


if __name__ == "__main__":
    main()
