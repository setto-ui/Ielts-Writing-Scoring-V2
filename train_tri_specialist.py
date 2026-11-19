"""
train_tri_specialist.py — Train and Save TriSpecialistModel
Uses cached features for instant training, then evaluates full pipeline on val.csv.
"""
import sys, json, pickle, warnings
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from scorer.two_stage_model import TriSpecialistModel
from evaluate import FEATURE_KEYS, _load_dataset, extract_all_features, features_to_array
from scorer.normalizer import normalize_features
from scorer.scorer import score_essay, round_to_ielts_band
from scorer.constraint import apply_constraints
from scorer.calibration import calibrate

MODEL_OUT = Path("data/model_tri_specialist.pkl")
CACHE_PATH = Path("data/cache_features_31.npz")

print("=" * 65)
print("  TRAINING TRI-SPECIALIST TWO-STAGE MODEL")
print("=" * 65)

# 1. Load train features from cache
data = np.load(CACHE_PATH)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]

print(f"Loaded: X_train {X_train.shape}, y_train {y_train.shape}")

# 2. Train model
model = TriSpecialistModel(alpha_low=0.6, alpha_high=0.6)
model.train(X_train, y_train, FEATURE_KEYS)
model.save(MODEL_OUT)
print(f"Saved model to {MODEL_OUT}")

# 3. Evaluate full pipeline on Val Set (190 samples)
val_data = _load_dataset(Path("data/val.csv"))
print(f"\nEvaluating Full Pipeline with Tri-Specialist on Val Set ({len(val_data)} samples)...")

preds_tri, trues = [], []
preds_base = []

# Load baseline model for direct A/B comparison
from scorer.ml_model import IELTSModel
base_model = IELTSModel.load("data/model_exp_6JB.pkl")

for sample in val_data:
    essay = sample.get("essay", "")
    prompt = sample.get("prompt", "")
    true_b = float(sample.get("band", 5.0))
    if not essay.strip() or not prompt.strip():
        continue
    
    raw_feat = extract_all_features(essay, prompt)
    norm_feat = normalize_features(raw_feat)
    arr = features_to_array(norm_feat, FEATURE_KEYS).reshape(1, -1)
    
    # 1. Tri-Specialist Prediction through full pipeline
    ml_preds_tri = model.predict(arr)
    score_res_tri = score_essay(
        normalized_features=norm_feat,
        raw_features=raw_feat,
        ml_predictions=ml_preds_tri,
    )
    gated_tri = score_res_tri["scores"]
    constrained_tri, _ = apply_constraints(gated_tri, raw_feat)
    final_tri = calibrate(constrained_tri.get("overall", 5.0))
    
    # 2. Baseline Prediction through full pipeline
    ml_preds_base = base_model.predict(arr)
    score_res_base = score_essay(
        normalized_features=norm_feat,
        raw_features=raw_feat,
        ml_predictions=ml_preds_base,
    )
    gated_base = score_res_base["scores"]
    constrained_base, _ = apply_constraints(gated_base, raw_feat)
    final_base = calibrate(constrained_base.get("overall", 5.0))

    preds_tri.append(final_tri)
    preds_base.append(final_base)
    trues.append(true_b)

preds_tri = np.array(preds_tri)
preds_base = np.array(preds_base)
trues = np.array(trues)

def print_metrics(name, yp, yt):
    mae = mean_absolute_error(yt, yp)
    acc05 = np.mean(np.abs(yp - yt) <= 0.5) * 100
    bias = np.mean(yp - yt)
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    print(f"  Overall MAE: {mae:.3f} | Bias: {bias:+.3f} | +-0.5 Acc: {acc05:.1f}%")
    for gname, lo, hi in [("Low  (4.0-5.5)", 4.0, 5.5), ("Mid  (6.0-7.0)", 6.0, 7.0), ("High (7.5-9.0)", 7.5, 9.0)]:
        idx = (yt >= lo) & (yt <= hi)
        if np.sum(idx) > 0:
            g_mae = mean_absolute_error(yt[idx], yp[idx])
            g_acc05 = np.mean(np.abs(yp[idx] - yt[idx]) <= 0.5) * 100
            g_bias = np.mean(yp[idx] - yt[idx])
            print(f"    {gname:<16} (N={np.sum(idx):>3}): MAE={g_mae:.3f} | Bias={g_bias:+.3f} | +-0.5={g_acc05:>5.1f}%")

print_metrics("Baseline Model 6J-B (Full Pipeline)", preds_base, trues)
print_metrics("Tri-Specialist Two-Stage Model (Full Pipeline)", preds_tri, trues)
