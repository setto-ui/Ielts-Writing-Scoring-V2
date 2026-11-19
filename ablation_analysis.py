import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from scorer.ml_model import IELTSModel
from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import normalize_features
from scorer.scorer import score_essay, round_to_ielts_band
from scorer.constraint import apply_constraints
from scorer.calibration import calibrate

def get_metrics(true_bands, pred_bands):
    errors = np.abs(np.array(pred_bands) - np.array(true_bands))
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    bias = np.mean(np.array(pred_bands) - np.array(true_bands))
    acc_05 = np.mean(errors <= 0.5) * 100
    return mae, rmse, bias, acc_05

def print_band_metrics(title, true_bands, pred_bands):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    mae, rmse, bias, acc_05 = get_metrics(true_bands, pred_bands)
    print(f"  Overall MAE:      {mae:.3f}")
    print(f"  Overall RMSE:     {rmse:.3f}")
    print(f"  Overall Bias:     {bias:.3f}")
    print(f"  ±0.5 Accuracy:    {acc_05:.1f}%")
    
    print("\n  Per-band Metrics:")
    print(f"  {'Band':>5} | {'N':>4} | {'MAE':>5} | {'Bias':>6} | {'±0.5 Acc':>8}")
    print("  " + "-" * 42)
    
    bands_dict = {}
    for t, p in zip(true_bands, pred_bands):
        if t not in bands_dict:
            bands_dict[t] = []
        bands_dict[t].append(p)
        
    for b in sorted(bands_dict.keys()):
        preds = bands_dict[b]
        n = len(preds)
        errs = [abs(p - b) for p in preds]
        b_mae = np.mean(errs)
        b_bias = np.mean([p - b for p in preds])
        b_acc = sum(1 for e in errs if e <= 0.5) / n * 100
        print(f"  {b:>5.1f} | {n:>4} | {b_mae:>5.3f} | {b_bias:>6.3f} | {b_acc:>7.1f}%")

def main():
    dataset_path = Path("data/val.csv")
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        sys.exit(1)

    model_path = "data/ml_model.pkl"
    model = IELTSModel.load(model_path)

    dataset = _load_dataset(dataset_path)
    
    true_bands = []
    
    # A. Raw LightGBM (rounded to nearest 0.5)
    preds_raw = []
    
    # B. Raw + Constraint (which includes hard gates)
    preds_constrained = []
    
    # C. Raw + Constraint + Isotonic
    preds_calibrated = []

    print(f"Running ablation on {len(dataset)} samples from val.csv...")
    
    for i, sample in enumerate(dataset):
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        true_band = sample.get("band", 5.0)

        if not essay.strip() or not prompt.strip():
            continue

        raw_features = extract_all_features(essay, prompt)
        norm_features = normalize_features(raw_features)

        X = features_to_array(norm_features, FEATURE_KEYS).reshape(1, -1)
        ml_preds = model.predict(X)
        
        raw_overall = ml_preds["overall"]
        
        # Hard gates + subscore estimation
        score_result = score_essay(
            normalized_features=norm_features,
            raw_features=raw_features,
            ml_predictions=ml_preds,
        )
        gated_scores = score_result["scores"]
        
        # Proportional Constraints
        constrained_scores, _ = apply_constraints(gated_scores, raw_features)
        c_overall = constrained_scores.get("overall")
        
        # Calibrate
        calibrated_overall = calibrate(c_overall)
        
        true_bands.append(true_band)
        preds_raw.append(round_to_ielts_band(float(raw_overall)))
        preds_constrained.append(round_to_ielts_band(float(c_overall)))
        preds_calibrated.append(calibrated_overall)

    print_band_metrics("A. Raw LightGBM (Rounded)", true_bands, preds_raw)
    print_band_metrics("B. Raw + Constraints", true_bands, preds_constrained)
    print_band_metrics("C. Raw + Constraints + Isotonic Calibration", true_bands, preds_calibrated)

if __name__ == "__main__":
    main()
