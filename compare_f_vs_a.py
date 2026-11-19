"""
compare_f_vs_a.py - Compare Model F (Enhanced Features) vs Model A (Baseline)
on val.csv (190 samples). Both models use L2, No Weight, No Constraint, No Isotonic.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from scorer.ml_model import IELTSModel
from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import normalize_features
from scorer.scorer import round_to_ielts_band

# Model A used old 21 features — load its original FEATURE_KEYS
FEATURE_KEYS_A = sorted([
    "argument_depth_score", "awl_ratio", "complex_clause_ratio",
    "connectors_per_sentence", "discourse_marker_count",
    "discourse_marker_variety", "gaming_penalty", "grammar_error_rate",
    "has_conclusion", "has_intro", "mattr", "max_word_repetition_ratio",
    "paragraph_count", "rare_word_ratio", "semantic_coherence",
    "sent_length_variance", "sentence_count", "task_coverage_score",
    "tense_consistent", "topic_similarity", "word_count",
])

def get_metrics(true_bands, pred_bands):
    errors = np.abs(np.array(pred_bands) - np.array(true_bands))
    return {
        "mae": np.mean(errors),
        "rmse": np.sqrt(np.mean(errors ** 2)),
        "bias": np.mean(np.array(pred_bands) - np.array(true_bands)),
        "acc05": np.mean(errors <= 0.5) * 100,
    }

def print_model_report(name, true_bands, pred_bands):
    m = get_metrics(true_bands, pred_bands)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Overall MAE:      {m['mae']:.3f}")
    print(f"  Overall RMSE:     {m['rmse']:.3f}")
    print(f"  Overall Bias:     {m['bias']:.3f}")
    print(f"  ±0.5 Accuracy:    {m['acc05']:.1f}%")

    low_idx  = [i for i, b in enumerate(true_bands) if 4.0 <= b <= 5.5]
    mid_idx  = [i for i, b in enumerate(true_bands) if 6.0 <= b <= 7.0]
    high_idx = [i for i, b in enumerate(true_bands) if 7.5 <= b <= 9.0]

    def group_line(label, idxs):
        if not idxs:
            return
        tb = [true_bands[i] for i in idxs]
        pb = [pred_bands[i] for i in idxs]
        gm = get_metrics(tb, pb)
        print(f"  {label:15s} | N: {len(idxs):>3} | MAE: {gm['mae']:.3f} | Bias: {gm['bias']:>+6.3f} | ±0.5: {gm['acc05']:5.1f}%")

    print()
    group_line("Low  (4.0-5.5)", low_idx)
    group_line("Mid  (6.0-7.0)", mid_idx)
    group_line("High (7.5-9.0)", high_idx)

    print(f"\n  {'Band':>5} | {'N':>4} | {'MAE':>5} | {'Bias':>6} | {'±0.5':>5}")
    print("  " + "-" * 38)
    bands_dict: dict = {}
    for t, p in zip(true_bands, pred_bands):
        bands_dict.setdefault(t, []).append(p)
    for b in sorted(bands_dict.keys()):
        preds = bands_dict[b]
        n = len(preds)
        errs = [abs(p - b) for p in preds]
        b_mae = np.mean(errs)
        b_bias = np.mean([p - b for p in preds])
        b_acc = sum(1 for e in errs if e <= 0.5) / n * 100
        print(f"  {b:>5.1f} | {n:>4} | {b_mae:>5.3f} | {b_bias:>+6.3f} | {b_acc:>4.0f}%")

    # Feature Importance (Model F only — has lgbm model)
    return m

def run_model(model, feat_keys, dataset):
    true_bands, pred_bands = [], []
    for sample in dataset:
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        true_band = sample.get("band", 5.0)
        if not essay.strip() or not prompt.strip():
            continue
        raw_feat = extract_all_features(essay, prompt)
        norm_feat = normalize_features(raw_feat)
        X = features_to_array(norm_feat, feat_keys).reshape(1, -1)
        raw_overall = float(model.predict(X)["overall"])
        true_bands.append(true_band)
        pred_bands.append(round_to_ielts_band(raw_overall))
    return true_bands, pred_bands

def print_feature_importance(model):
    lgb_model = model.models.get("overall")
    if not lgb_model:
        return
    gain = lgb_model.booster_.feature_importance(importance_type='gain')
    names = FEATURE_KEYS
    imp = sorted(zip(names, gain), key=lambda x: -x[1])
    print("\n  Feature Importance (Model F — Gain):")
    print(f"  {'Feature':>25} | {'Gain':>8}")
    print("  " + "-" * 38)
    for feat, g in imp[:15]:
        print(f"  {feat:>25} | {g:>8.1f}")

def main():
    val_path = Path("data/val.csv")
    print(f"Loading {val_path}...")
    dataset = _load_dataset(val_path)

    model_a_path = Path("data/model_exp_A.pkl")
    model_f_path = Path("data/model_exp_F.pkl")

    for p in [model_a_path, model_f_path]:
        if not p.exists():
            print(f"[ERROR] {p} not found. Run training scripts first.")
            sys.exit(1)

    model_a = IELTSModel.load(model_a_path)
    model_f = IELTSModel.load(model_f_path)

    print("Evaluating Model A (Baseline, 21 features)...")
    true_a, pred_a = run_model(model_a, FEATURE_KEYS_A, dataset)
    ma = print_model_report("Model A — Baseline (L2, 21 features)", true_a, pred_a)

    print("\nEvaluating Model F (Enhanced, 27 features)...")
    true_f, pred_f = run_model(model_f, FEATURE_KEYS, dataset)
    mf = print_model_report("Model F — Enhanced (L2, 27 features)", true_f, pred_f)

    # Print feature importance of Model F
    print_feature_importance(model_f)

    # Delta summary
    print(f"\n{'='*60}")
    print("  DELTA SUMMARY (F - A)")
    print(f"{'='*60}")
    print(f"  Overall MAE : {mf['mae'] - ma['mae']:>+.3f}")
    print(f"  Overall RMSE: {mf['rmse'] - ma['rmse']:>+.3f}")
    print(f"  ±0.5 Acc    : {mf['acc05'] - ma['acc05']:>+.1f}%")
    print(f"  Overall Bias: {mf['bias'] - ma['bias']:>+.3f}")

if __name__ == "__main__":
    main()
