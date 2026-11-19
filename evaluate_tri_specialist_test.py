"""
evaluate_tri_specialist_test.py — Final Verification of TriSpecialistModel on Locked Test Set (N=191)
Rules: Read-only on test.csv. Compares Baseline 6J-B vs TriSpecialist.
"""
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from scorer.two_stage_model import TriSpecialistModel
from scorer.ml_model import IELTSModel
from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import normalize_features
from scorer.calibration import calibrate
from scorer.scorer import round_to_ielts_band

print("=" * 65)
print("  FINAL BENCHMARK ON LOCKED TEST SET (N=191)")
print("=" * 65)

test_data = _load_dataset(Path("data/test.csv"))
print(f"Loaded {len(test_data)} test samples.")

tri_model = TriSpecialistModel.load("data/model_tri_specialist.pkl")
base_model = IELTSModel.load("data/model_exp_6JB.pkl")

TEST_CACHE = Path("data/cache_features_31_test.npz")

if TEST_CACHE.exists():
    print(f"Loading pre-computed test features from {TEST_CACHE}...")
    tc = np.load(TEST_CACHE)
    X_test = tc["X_test"]
    y_true = tc["y_true"]
else:
    print("Extracting test features...")
    X_list, y_list = [], []
    for sample in tqdm(test_data):
        essay = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        band = float(sample.get("band", 5.0))
        if not essay.strip() or not prompt.strip():
            continue
        raw_feat = extract_all_features(essay, prompt)
        norm_feat = normalize_features(raw_feat)
        arr = features_to_array(norm_feat, FEATURE_KEYS)
        X_list.append(arr)
        y_list.append(band)
    X_test = np.vstack(X_list)
    y_true = np.array(y_list)
    np.savez_compressed(TEST_CACHE, X_test=X_test, y_true=y_true)
    print(f"Cached test features to {TEST_CACHE}.")

# 1. Baseline: IELTSModel + Calibrator
base_raw = base_model.predict(X_test)["overall"]
preds_base = np.array([calibrate(r) for r in base_raw])

# 2. Tri-Specialist: Native Soft Blending + Rounding
tri_raw = tri_model.predict(X_test)["overall"]
preds_tri = np.array([round_to_ielts_band(float(r)) for r in tri_raw])

def print_report(name, yp, yt):
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(np.mean((yp - yt) ** 2))
    bias = np.mean(yp - yt)
    acc05 = np.mean(np.abs(yp - yt) <= 0.5) * 100
    acc10 = np.mean(np.abs(yp - yt) <= 1.0) * 100
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    print(f"  Overall MAE:      {mae:.3f}")
    print(f"  Overall RMSE:     {rmse:.3f}")
    print(f"  Overall Bias:     {bias:+.3f}")
    print(f"  +-0.5 Accuracy:   {acc05:.1f}%")
    print(f"  +-1.0 Accuracy:   {acc10:.1f}%\n")
    print(f"  Per-Group Breakdown:")
    for gname, lo, hi in [("Low  (4.0-5.5)", 4.0, 5.5), ("Mid  (6.0-7.0)", 6.0, 7.0), ("High (7.5-9.0)", 7.5, 9.0)]:
        idx = (yt >= lo) & (yt <= hi)
        if np.sum(idx) > 0:
            g_mae = mean_absolute_error(yt[idx], yp[idx])
            g_bias = np.mean(yp[idx] - yt[idx])
            g_acc05 = np.mean(np.abs(yp[idx] - yt[idx]) <= 0.5) * 100
            print(f"    {gname:<16} (N={np.sum(idx):>3}): MAE={g_mae:.3f} | Bias={g_bias:+.3f} | +-0.5={g_acc05:>5.1f}%")

print_report("Production Baseline Model 6J-B (Locked Test)", preds_base, y_true)
print_report("Tri-Specialist Two-Stage Model (Locked Test)", preds_tri, y_true)
