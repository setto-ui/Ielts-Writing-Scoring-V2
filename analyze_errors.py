"""
analyze_errors.py — Systematic Error Analysis for IELTS Writing Scorer v2.
Calculates:
1. MAE / Accuracy comparison: Before vs After Constraint Layer.
2. Error histogram.
3. Confusion matrix.
4. Top 50 largest error essays details (features & constraints triggered).
"""

import csv
import json
import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

from scorer.features_text import extract_text_features
from scorer.features_essay import extract_essay_features
from scorer.features_task import extract_task_features
from scorer.features_antigaming import extract_antigaming_features
from scorer.normalizer import normalize_features
from scorer.ml_model import IELTSModel
from scorer.scorer import score_essay, round_to_ielts_band
from scorer.constraint import apply_constraints
from scorer.calibration import calibrate

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

def _load_dataset(path: Path) -> list:
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
                dataset.append({
                    "essay": row.get("Essay", "").strip(),
                    "prompt": row.get("Question", "").strip(),
                    "band": band,
                })
        return dataset
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def extract_all_features(essay: str, prompt: str) -> dict:
    text_feat = extract_text_features(essay)
    essay_feat = extract_essay_features(essay, prompt)
    task_feat = extract_task_features(essay, prompt)
    antigaming_feat = extract_antigaming_features(essay, prompt)

    merged = {}
    merged["mattr"] = text_feat.get("mattr", 0.0)
    merged["rare_word_ratio"] = text_feat.get("rare_word_ratio", 0.0)
    merged["awl_ratio"] = text_feat.get("awl_ratio", 0.0)
    merged["grammar_error_rate"] = text_feat.get("grammar_error_rate", 0.0)
    merged["complex_clause_ratio"] = text_feat.get("complex_clause_ratio", 0.0)
    merged["sent_length_variance"] = text_feat.get("sent_length_variance", 0.0)
    merged["tense_consistent"] = 1.0 if text_feat.get("tense_consistent", True) else 0.0
    merged["max_word_repetition_ratio"] = text_feat.get("max_word_repetition_ratio", 0.0)
    merged["paragraph_count"] = float(essay_feat.get("paragraph_count", 1))
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
    merged["connectors_per_sentence"] = antigaming_feat.get("connectors_per_sentence", 0.0)
    merged["gaming_penalty"] = antigaming_feat.get("gaming_penalty", 0.0)

    # Flags for error details
    merged["_gates_fired"] = []
    return merged

def features_to_array(features: dict) -> np.ndarray:
    return np.array([features.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float64)

def run_pipelines(essay: str, prompt: str, model: IELTSModel):
    # 1. Extract
    raw_feats = extract_all_features(essay, prompt)
    # 2. Normalize
    norm_feats = normalize_features(raw_feats)
    # 3. Predict
    X = features_to_array(norm_feats).reshape(1, -1)
    ml_preds = model.predict(X)
    ml_scores = {k: float(v[0]) if hasattr(v, '__len__') else float(v) for k, v in ml_preds.items()}

    # 4. Gating
    score_res = score_essay(norm_feats, raw_feats, ml_scores)
    gated_scores = score_res["scores"]
    gates_fired = score_res["gates"]

    # --- PIPELINE A: Before Constraints ---
    raw_overall_a = gated_scores.get("overall")
    band_before = calibrate(raw_overall_a)

    # --- PIPELINE B: After Constraints ---
    constrained_scores, constraints_applied = apply_constraints(gated_scores, raw_feats)
    raw_overall_b = constrained_scores.get("overall")
    band_after = calibrate(raw_overall_b)

    return {
        "raw_features": raw_feats,
        "ml_scores": ml_scores,
        "gated_scores": gated_scores,
        "gates_fired": gates_fired,
        "band_before": band_before,
        "constrained_scores": constrained_scores,
        "constraints_applied": constraints_applied,
        "band_after": band_after
    }

def main():
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description="Error analysis tool")
    parser.add_argument("--dataset", default="data/ielts_writing_dataset.csv")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset {dataset_path} not found.")
        sys.exit(1)

    try:
        model = IELTSModel.load("data/ml_model.pkl")
    except FileNotFoundError:
        print("[ERROR] Model ml_model.pkl not found. Train first.")
        sys.exit(1)

    print("Loading dataset...")
    dataset = _load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} essays. Starting pipeline runs...")

    results = []
    for i, sample in enumerate(dataset):
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        true_band = sample.get("band", 5.0)

        if not essay.strip() or not prompt.strip():
            continue

        try:
            res = run_pipelines(essay, prompt, model)
            res["true_band"] = true_band
            res["index"] = i + 1
            res["essay_text"] = essay
            res["prompt_text"] = prompt
            results.append(res)
            if (len(results)) % 200 == 0:
                print(f"  Processed {len(results)}/{len(dataset)}...")
        except Exception as e:
            print(f"  [ERROR] at sample {i+1}: {e}")

    total_samples = len(results)
    if total_samples == 0:
        print("[ERROR] No essays processed.")
        sys.exit(1)

    # 1. Compare metrics
    errors_before = [abs(r["band_before"] - r["true_band"]) for r in results]
    errors_after = [abs(r["band_after"] - r["true_band"]) for r in results]

    mae_before = np.mean(errors_before)
    mae_after = np.mean(errors_after)
    acc_before = sum(1 for e in errors_before if e <= 0.5) / total_samples * 100
    acc_after = sum(1 for e in errors_after if e <= 0.5) / total_samples * 100

    print("\n" + "=" * 60)
    print("  METRICS COMPARISON: BEFORE VS AFTER CONSTRAINTS")
    print("=" * 60)
    print(f"  Metric              Before Constraints    After Constraints")
    print(f"  -----------------------------------------------------------")
    print(f"  MAE                 {mae_before:.4f}               {mae_after:.4f}")
    print(f"  Acc (<= 0.5)        {acc_before:.2f}%               {acc_after:.2f}%")
    diff_mae = mae_after - mae_before
    if diff_mae > 0:
        print(f"  --> Constraint Layer DEGRADED MAE by +{diff_mae:.4f} (hurt model)")
    else:
        print(f"  --> Constraint Layer IMPROVED MAE by {diff_mae:.4f} (helped model)")

    # 2. Error Histogram (After Constraints)
    hist = Counter(errors_after)
    print("\n" + "=" * 60)
    print("  ERROR HISTOGRAM (AFTER CONSTRAINTS)")
    print("=" * 60)
    print(f"  Abs Error    Count    Percentage")
    print(f"  ---------------------------------")
    for err_val in sorted(hist.keys()):
        count = hist[err_val]
        pct = count / total_samples * 100
        print(f"  {err_val:>9.1f}    {count:>5}    {pct:>9.2f}%")

    # 3. Confusion Matrix (After Constraints)
    bands_list = sorted(list(set([r["true_band"] for r in results] + [r["band_after"] for r in results])))
    matrix = defaultdict(lambda: defaultdict(int))
    for r in results:
        matrix[r["true_band"]][r["band_after"]] += 1

    print("\n" + "=" * 60)
    print("  CONFUSION MATRIX (True vs Pred)")
    print("=" * 60)
    # Header
    header_str = "  True \\ Pred | " + "  ".join(f"{b:3.1f}" for b in bands_list)
    print(header_str)
    print("  " + "-" * len(header_str))
    for t in bands_list:
        row_str = f"     {t:3.1f}      | " + "  ".join(f"{matrix[t][p]:3d}" if matrix[t][p] > 0 else "  - " for p in bands_list)
        print(row_str)

    # 4. Top 50 Largest Errors Detail
    sorted_results = sorted(results, key=lambda x: abs(x["band_after"] - x["true_band"]), reverse=True)
    
    print("\n" + "=" * 60)
    print("  TOP 50 LARGEST ERROR SAMPLES")
    print("=" * 60)
    
    # Save detailed report to file
    report_path = Path("data/error_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("IELTS Writing Scorer v2 — Error Analysis Report\n")
        f.write("================================================\n\n")
        f.write(f"Total processed samples: {total_samples}\n")
        f.write(f"MAE Before Constraints: {mae_before:.4f} | After: {mae_after:.4f}\n")
        f.write(f"Accuracy (<=0.5) Before: {acc_before:.2f}% | After: {acc_after:.2f}%\n\n")
        
        f.write("TOP 50 LARGEST ERROR SAMPLES DETAIL\n")
        f.write("-----------------------------------\n\n")
        
        for idx, r in enumerate(sorted_results[:50]):
            err = abs(r["band_after"] - r["true_band"])
            f.write(f"Rank {idx+1} | Sample Index: {r['index']} | True Band: {r['true_band']:.1f} | Pred Band: {r['band_after']:.1f} (Before constraint: {r['band_before']:.1f}) | Error: {err:.1f}\n")
            f.write(f"  Essay Length (words): {r['raw_features']['word_count']:.0f}\n")
            f.write(f"  Topic Similarity: {r['raw_features']['topic_similarity']:.3f}\n")
            f.write(f"  Gaming Penalty: {r['raw_features']['gaming_penalty']:.3f}\n")
            
            # Print Gated vs Constrained Sub-scores
            f.write("  Criterion Sub-scores (ML Predicted -> Gated -> Constrained):\n")
            for c_name in ["task_response", "lexical", "grammar", "coherence"]:
                ml_s = r["ml_scores"].get(c_name, 0.0)
                gate_s = r["gated_scores"].get(c_name, 0.0)
                const_s = r["constrained_scores"].get(c_name, 0.0)
                f.write(f"    - {c_name:<15}: ML {ml_s:.2f} -> Gated {gate_s:.2f} -> Constrained {const_s:.2f}\n")
            
            # Print Rules
            f.write(f"  Gates fired: {r['gates_fired']}\n")
            f.write(f"  Constraints applied: {r['constraints_applied']}\n")
            f.write("  Essay Clip: " + r["essay_text"][:200].replace('\n', ' ') + "...\n")
            f.write("-" * 80 + "\n\n")

    print(f"Detailed Top 50 report saved to: {report_path.resolve()}")
    
    # Print summary of top 10 on screen
    print(f"  {'Rank':<5} {'Idx':<5} {'True':<5} {'Before':<7} {'After':<6} {'Err':<5} {'Applied Constraints'}")
    print(f"  ---------------------------------------------------------------------------------")
    for idx, r in enumerate(sorted_results[:15]):
        err = abs(r["band_after"] - r["true_band"])
        constraints_str = str(r['constraints_applied']).encode('ascii', 'replace').decode('ascii')
        print(f"  {idx+1:<5} {r['index']:<5} {r['true_band']:<5.1f} {r['band_before']:<7.1f} {r['band_after']:<6.1f} {err:<5.1f} {constraints_str}")

if __name__ == "__main__":
    main()
