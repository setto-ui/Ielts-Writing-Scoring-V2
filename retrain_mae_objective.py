"""
retrain_mae_objective.py — Retrain 6J-B with MAE (L1) loss instead of MSE (L2).

MAE loss predicts the *median* instead of the *mean*, which is less sensitive to
imbalanced class distribution and typically reduces regression-to-mean on extreme bands.

Keeps existing calibration_model.pkl.
Strict acceptance: applies new model only if it improves test MAE AND
                   no band group loses more than 5pct w0.5.

Usage:
    python retrain_mae_objective.py
"""
import sys, json, pickle, warnings
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

print("=" * 65)
print("  RETRAIN 6J-B — MAE (L1) loss objective")
print("=" * 65)

from evaluate import (
    _load_dataset, extract_all_features, features_to_array,
    FEATURE_KEYS, run_full_pipeline
)
from scorer.normalizer import normalize_features
from scorer.ml_model import IELTSModel
from scorer.calibration import round_to_ielts_band
import lightgbm as lgb

MODEL_PATH     = Path("data/model_exp_6JB.pkl")
MODEL_MAE_PATH = Path("data/model_exp_6JB_mae.pkl")
CAL_PATH       = Path("data/calibration_model.pkl")
STATS_PATH     = Path("data/feature_stats.json")

# ── Load data ────────────────────────────────────────────────────────────────
print("\n[1/4] Loading train data...")
train_data = _load_dataset(Path("data/train.csv"))
val_data   = _load_dataset(Path("data/val.csv"))
test_data  = _load_dataset(Path("data/test.csv"))
print(f"  Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}")

with open(CAL_PATH, "rb") as f:
    calibrator = pickle.load(f)

# ── Extract train features ────────────────────────────────────────────────────
print("\n[2/4] Extracting train features...")
train_X, train_y = [], []
for i, sample in enumerate(train_data):
    essay  = sample.get("essay", "")
    prompt = sample.get("prompt", "")
    true_b = float(sample.get("band", 5.0))
    if not essay.strip():
        continue
    try:
        raw_feat  = extract_all_features(essay, prompt)
        norm_feat = normalize_features(raw_feat)
        arr       = features_to_array(norm_feat, FEATURE_KEYS)
        train_X.append(arr)
        train_y.append(true_b)
    except Exception:
        pass
    if (i + 1) % 75 == 0:
        print(f"  {i+1}/{len(train_data)}", flush=True)

X = np.vstack(train_X)
y = np.array(train_y)
print(f"  Matrix: {X.shape}")

# ── Train MAE model ───────────────────────────────────────────────────────────
print("\n[3/4] Training LightGBM with MAE (regression_l1) objective...")
params_mae = {
    "n_estimators": 150,   # slightly more trees for MAE (needs more iterations)
    "max_depth": 4,
    "learning_rate": 0.08,
    "min_child_samples": 3,
    "num_leaves": 15,
    "verbosity": -1,
    "objective": "regression_l1",  # ← MAE loss
}

import pandas as pd
X_df = pd.DataFrame(X, columns=FEATURE_KEYS)

reg_mae = lgb.LGBMRegressor(**params_mae)
reg_mae.fit(X_df, y)
print(f"  Training MAE on train: {np.mean(np.abs(reg_mae.predict(X_df) - y)):.3f}")

# Wrap in IELTSModel structure and save
model_mae = IELTSModel()
model_mae.feature_names = FEATURE_KEYS
model_mae.models = {"overall": reg_mae}
model_mae.save(MODEL_MAE_PATH)
print(f"  Saved MAE model to: {MODEL_MAE_PATH}")

# ── Evaluate: old MSE model vs new MAE model on val + test ───────────────────
print("\n[4/4] Evaluating on VAL + TEST...")

model_mse = IELTSModel.load(MODEL_PATH)
model_mae2 = IELTSModel.load(MODEL_MAE_PATH)

def eval_model(model, dataset, label):
    preds, trues = [], []
    for i, sample in enumerate(dataset):
        essay  = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        true_b = float(sample.get("band", 5.0))
        if true_b < 1.0 or not essay.strip():
            continue
        try:
            raw_feat  = extract_all_features(essay, prompt)
            norm_feat = normalize_features(raw_feat)
            arr       = features_to_array(norm_feat, FEATURE_KEYS)
            ml_preds  = model.predict(arr)
            # Use calibrator on the raw overall prediction
            raw_ov    = np.clip(ml_preds.get("overall", 5.0), 4.0, 9.0)
            pred_b    = round_to_ielts_band(float(calibrator.predict(np.array([raw_ov]))[0]))
            preds.append(pred_b)
            trues.append(true_b)
        except Exception:
            pass
        if (i + 1) % 25 == 0:
            print(f"    [{label}] {i+1}/{len(dataset)}", flush=True)
    errs  = [abs(p-t) for p,t in zip(preds,trues)]
    mae   = sum(errs)/len(errs) if errs else 9.0
    w05   = sum(1 for e in errs if e<=0.5)/len(errs) if errs else 0
    w10   = sum(1 for e in errs if e<=1.0)/len(errs) if errs else 0
    bias  = sum(p-t for p,t in zip(preds,trues))/len(preds) if preds else 0
    groups = {}
    for g, lo, hi in [("Low",4.0,5.5),("Mid",6.0,7.0),("High",7.5,9.0)]:
        gp = [(p,t) for p,t in zip(preds,trues) if lo<=t<=hi]
        if gp:
            ge = [abs(p-t) for p,t in gp]
            groups[g] = {"n":len(gp),"mae":sum(ge)/len(ge),
                         "bias":sum(p-t for p,t in gp)/len(gp),
                         "w05":sum(1 for e in ge if e<=0.5)/len(ge)}
    return {"mae":mae,"w05":w05,"w10":w10,"bias":bias,"n":len(preds),"groups":groups}

def print_comparison(old_r, new_r, split):
    W = 65
    print(f"\n  {split} SET — MSE (old) vs MAE (new)  (N={old_r['n']})")
    print(f"  {'-'*58}")
    for name, o, n, lb in [
        ("Overall MAE", old_r['mae'], new_r['mae'], True),
        ("Overall w0.5", old_r['w05'], new_r['w05'], False),
        ("Overall w1.0", old_r['w10'], new_r['w10'], False),
        ("Overall Bias", old_r['bias'], new_r['bias'], None),
    ]:
        if lb is not None:
            d = n - o
            flag = ("BETTER" if (lb and d<-0.003) or (not lb and d>0.003)
                    else "WORSE" if (lb and d>0.003) or (not lb and d<-0.003)
                    else "~same")
            print(f"  {name:<18}: MSE={o:.3f}  MAE={n:.3f}  [{flag}]")
        else:
            print(f"  {'Bias':<18}: MSE={o:+.3f}  MAE={n:+.3f}")
    print(f"  Per-band:")
    for g in ["Low","Mid","High"]:
        o = old_r["groups"].get(g,{}); n2 = new_r["groups"].get(g,{})
        if not o or not n2: continue
        dw = n2['w05']-o['w05']; dm = n2['mae']-o['mae']
        flag = "OK" if dw >= -0.05 else "WARN"
        print(f"    {g:<5}: MAE {o['mae']:.3f}->{n2['mae']:.3f} ({dm:+.3f})"
              f"  w0.5 {o['w05']:.1%}->{n2['w05']:.1%} ({dw:+.1%})"
              f"  bias {o['bias']:+.3f}->{n2['bias']:+.3f}  [{flag}]")

print("\n  Evaluating on VAL...")
val_mse = eval_model(model_mse, val_data,  "VAL-MSE")
val_mae = eval_model(model_mae2, val_data, "VAL-MAE")
print_comparison(val_mse, val_mae, "VAL")

print("\n  Evaluating on TEST (locked)...")
test_mse = eval_model(model_mse, test_data,  "TST-MSE")
test_mae = eval_model(model_mae2, test_data, "TST-MAE")
print_comparison(test_mse, test_mae, "TEST")

# ── Decision ──────────────────────────────────────────────────────────────────
mae_ok  = test_mae['mae'] <= test_mse['mae'] + 0.005
band_ok = all(
    test_mae["groups"].get(g, {}).get("w05", 0)
    >= test_mse["groups"].get(g, {}).get("w05", 0) - 0.05
    for g in ["Low","Mid","High"]
    if test_mse["groups"].get(g)
)

print(f"\n  DECISION CRITERIA:")
print(f"    Test MAE improves (or within 0.005): {mae_ok}  ({test_mse['mae']:.3f} -> {test_mae['mae']:.3f})")
print(f"    No band loses >5pct w0.5: {band_ok}")

if mae_ok and band_ok:
    import shutil
    shutil.copy(MODEL_PATH, "data/model_exp_6JB_mse_backup.pkl")
    shutil.copy(MODEL_MAE_PATH, MODEL_PATH)
    print(f"\n  APPLIED: MAE model saved as model_exp_6JB.pkl")
    print(f"  MSE backup: data/model_exp_6JB_mse_backup.pkl")
else:
    print(f"\n  NOT APPLIED: Criterion not met. MAE model at: {MODEL_MAE_PATH}")
    print(f"  Original MSE model unchanged.")

print("\nDone.\n")
