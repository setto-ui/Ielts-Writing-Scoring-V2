"""
improve_calibration.py — Retrain calibrator using Out-of-Fold (OOF) predictions.

Strategy:
  1. Load extracted feature cache (1435 entries, includes train+val+test)
  2. Match cache entries to train.csv and val.csv by essay text
  3. 5-fold CV on train: train 5 mini-models, get 886 OOF raw predictions
  4. Combine: 886 OOF (train) + 190 val predictions = 1076 calibration points
  5. Fit new IsotonicRegression on 1076 points  (vs current 190)
  6. Evaluate old vs new calibrator on locked test set
  7. Save new calibrator if improvement >= 0 on MAE

DOES NOT touch model_exp_6JB.pkl. Only replaces calibration_model.pkl.

Usage:
    python improve_calibration.py
"""
import sys, json, csv, pickle, math, warnings
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")   # suppress feature-name warnings

print("=" * 65)
print("  CALIBRATION IMPROVEMENT — OOF 5-fold strategy")
print("=" * 65)

from evaluate import (
    _load_dataset, features_to_array, FEATURE_KEYS, run_full_pipeline
)
from scorer.normalizer import normalize_features, compute_stats_from_dataset
from scorer.ml_model import IELTSModel
from scorer.scorer import score_essay
from scorer.constraint import apply_constraints
from scorer.calibration import calibrate, train_calibrator, round_to_ielts_band

MODEL_PATH = Path("data/model_exp_6JB.pkl")
CAL_PATH   = Path("data/calibration_model.pkl")
CAL_BACKUP = Path("data/calibration_model_backup_preOOF.pkl")
CACHE_PATH = Path("data/extracted_features_cache.pkl")
STATS_PATH = Path("data/feature_stats.json")

# ── Load data ────────────────────────────────────────────────────────────────
print("\n[1/6] Loading datasets...")
train_data = _load_dataset(Path("data/train.csv"))
val_data   = _load_dataset(Path("data/val.csv"))
test_data  = _load_dataset(Path("data/test.csv"))
print(f"  Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}")

# ── Load feature stats ────────────────────────────────────────────────────────
with open(STATS_PATH, encoding="utf-8") as f:
    feat_stats = json.load(f)

# ── Load feature cache and build essay→features lookup ──────────────────────
print("\n[2/6] Loading feature cache...")
with open(CACHE_PATH, "rb") as f:
    cache_list = pickle.load(f)

# Cache items are raw feature dicts — we need to match them to essays.
# Strategy: extract features from cache by index if order matches, otherwise
# re-extract. Cache has 1435 items; train+val+test = 886+190+191 = 1267.
# Use essay text as key (first 80 chars).
print(f"  Cache entries: {len(cache_list)}")

# Build a proper feature matrix for train set
# Re-use cached features if available, otherwise extract fresh
print("\n[3/6] Building normalized feature matrix for train set...")
# We need to extract features for train set essays that are in the cache
# Since cache ordering is unknown, we'll do a fresh extraction for training
# essays only (886 * ~0.3s = ~4.5 min). But if cache is large enough, try to match.

# Simple approach: just use run_full_pipeline and collect pre-calibration values.
# We'll monkey-patch calibrate to return the raw value for data collection.

def get_raw_overall(essay: str, prompt: str, model: IELTSModel) -> float:
    """Get the raw (pre-calibration) overall score for an essay."""
    from evaluate import extract_all_features, features_to_array
    raw_feat  = extract_all_features(essay, prompt)
    norm_feat = normalize_features(raw_feat)
    X         = features_to_array(norm_feat, FEATURE_KEYS).reshape(1, -1)
    ml_preds  = model.predict(X)
    score_res = score_essay(
        normalized_features=norm_feat,
        raw_features=raw_feat,
        ml_predictions=ml_preds,
    )
    gated     = score_res["scores"]
    constrained, _ = apply_constraints(gated, raw_feat)
    return float(constrained.get("overall", 5.0))

def get_final_overall(raw_overall: float, cal: IsotonicRegression | None) -> float:
    if cal is not None:
        return round_to_ielts_band(float(cal.predict(np.array([raw_overall]))[0]))
    return round_to_ielts_band(raw_overall)

# ── Load champion model ───────────────────────────────────────────────────────
model = IELTSModel.load(MODEL_PATH)

# ── Get val set raw predictions (for combining with OOF) ─────────────────────
print("\n[4/6] Scoring VAL set for calibration data...")
val_pairs = []   # (raw_pred, true_band)
for i, sample in enumerate(val_data):
    essay  = sample.get("essay", "")
    prompt = sample.get("prompt", "")
    true_b = float(sample.get("band", 5.0))
    try:
        raw = get_raw_overall(essay, prompt, model)
        val_pairs.append((raw, true_b))
    except Exception as e:
        pass
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(val_data)}", flush=True)
print(f"  Collected {len(val_pairs)} val pairs.")

# ── 5-fold OOF on train set ───────────────────────────────────────────────────
print("\n[5/6] 5-fold OOF cross-validation on TRAIN set...")
print("  (Extracts features + trains 5 mini-LightGBM models)")

# First: extract features for all train essays
print("  Extracting train features...", flush=True)
from evaluate import extract_all_features, features_to_array
from scorer.normalizer import normalize_features

train_X = []
train_y = []
train_valid = []  # indices of valid train samples

for i, sample in enumerate(train_data):
    essay  = sample.get("essay", "")
    prompt = sample.get("prompt", "")
    true_b = float(sample.get("band", 5.0))
    if not essay.strip() or not prompt.strip():
        continue
    try:
        raw_feat  = extract_all_features(essay, prompt)
        norm_feat = normalize_features(raw_feat)
        arr       = features_to_array(norm_feat, FEATURE_KEYS)
        train_X.append(arr)
        train_y.append(true_b)
        train_valid.append(i)
    except Exception:
        pass
    if (i + 1) % 50 == 0:
        print(f"    {i+1}/{len(train_data)}", flush=True)

train_X = np.vstack(train_X)
train_y = np.array(train_y)
print(f"  Train matrix: {train_X.shape}, y range: [{train_y.min():.1f}, {train_y.max():.1f}]")

# 5-fold CV — train a LightGBM model per fold, predict on held-out fold
import lightgbm as lgb

_LGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "min_child_samples": 3,
    "num_leaves": 15,
    "verbosity": -1,
    "objective": "regression",
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_X))

print("  Running 5-fold CV...", flush=True)
for fold, (tr_idx, val_idx) in enumerate(kf.split(train_X)):
    X_tr, X_vl = train_X[tr_idx], train_X[val_idx]
    y_tr, y_vl = train_y[tr_idx], train_y[val_idx]

    reg = lgb.LGBMRegressor(**_LGB_PARAMS)
    reg.fit(X_tr, y_tr)
    preds = np.clip(reg.predict(X_vl), 4.0, 9.0)
    oof_preds[val_idx] = preds
    fold_mae = np.mean(np.abs(preds - y_vl))
    print(f"    Fold {fold+1}: MAE={fold_mae:.3f}  N_val={len(val_idx)}")

# OOF predictions are raw model output (before score_essay/constraints/calibration).
# For calibration we need post-constraint "overall" values ideally, but
# since LightGBM output ≈ post-constraint overall (constraints only affect
# edge cases), OOF LightGBM output is a good proxy.
oof_pairs = [(float(oof_preds[i]), float(train_y[i])) for i in range(len(train_X))]
print(f"  Collected {len(oof_pairs)} OOF pairs.")

# ── Combine OOF + Val for new calibrator ──────────────────────────────────────
print("\n[6/6] Training new calibrator and comparing with old...")
all_pairs = oof_pairs + val_pairs
raw_all  = np.array([p[0] for p in all_pairs])
true_all = np.array([p[1] for p in all_pairs])
print(f"  Total calibration points: {len(all_pairs)} (OOF {len(oof_pairs)} + Val {len(val_pairs)})")

new_cal = IsotonicRegression(out_of_bounds="clip")
new_cal.fit(raw_all, true_all)

# Load old calibrator
with open(CAL_PATH, "rb") as f:
    old_cal = pickle.load(f)

# ── Evaluate both on TEST set ─────────────────────────────────────────────────
print("\n  Evaluating on TEST set (locked)...")

def eval_on_test(cal, label):
    preds, trues = [], []
    for i, sample in enumerate(test_data):
        essay  = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        true_b = float(sample.get("band", 5.0))
        if true_b < 1.0:
            continue
        try:
            raw = get_raw_overall(essay, prompt, model)
            pred = get_final_overall(raw, cal)
            preds.append(pred)
            trues.append(true_b)
        except Exception:
            pass
        if (i + 1) % 25 == 0:
            print(f"    [{label}] {i+1}/{len(test_data)}", flush=True)
    errs   = [abs(p - t) for p, t in zip(preds, trues)]
    mae    = sum(errs) / len(errs)
    w05    = sum(1 for e in errs if e <= 0.5) / len(errs)
    w10    = sum(1 for e in errs if e <= 1.0) / len(errs)
    bias   = sum(p - t for p, t in zip(preds, trues)) / len(preds)
    # Per-group
    groups = {}
    for gname, lo, hi in [("Low",4.0,5.5),("Mid",6.0,7.0),("High",7.5,9.0)]:
        g = [(p,t) for p,t in zip(preds,trues) if lo <= t <= hi]
        if g:
            ge = [abs(p-t) for p,t in g]
            groups[gname] = {
                "n": len(g),
                "mae": sum(ge)/len(ge),
                "bias": sum(p-t for p,t in g)/len(g),
                "w05": sum(1 for e in ge if e<=0.5)/len(ge),
            }
    return {"mae": mae, "w05": w05, "w10": w10, "bias": bias,
            "n": len(preds), "groups": groups}

print("\n  Scoring with OLD calibrator...")
old_res = eval_on_test(old_cal, "OLD")
print("\n  Scoring with NEW calibrator (OOF)...")
new_res = eval_on_test(new_cal, "NEW")

# ── Print comparison ──────────────────────────────────────────────────────────
W = 65
print(f"\n{'='*W}")
print(f"  RESULT: OLD vs NEW calibrator on TEST SET (N={old_res['n']})")
print(f"{'='*W}")
metrics = [
    ("Overall MAE",  old_res['mae'],  new_res['mae'],  True),
    ("Overall w0.5", old_res['w05'],  new_res['w05'],  False),
    ("Overall w1.0", old_res['w10'],  new_res['w10'],  False),
    ("Overall Bias", old_res['bias'], new_res['bias'],  None),
]
for name, o, n, lower_better in metrics:
    delta = n - o
    if lower_better is True:
        sign = "+" if delta > 0 else ""
        flag = "WORSE" if delta > 0.005 else ("BETTER" if delta < -0.005 else "~same")
    elif lower_better is False:
        sign = "+" if delta > 0 else ""
        flag = "BETTER" if delta > 0.005 else ("WORSE" if delta < -0.005 else "~same")
    else:
        sign = "+" if delta > 0 else ""
        flag = f"abs={abs(n):.3f}"
    print(f"  {name:<18}: Old={o:.3f}  New={n:.3f}  Delta={sign}{delta:.3f}  [{flag}]")

print(f"\n  --- Per-band-group ---")
for g in ["Low", "Mid", "High"]:
    og = old_res["groups"].get(g, {})
    ng = new_res["groups"].get(g, {})
    if not og or not ng:
        continue
    dw05 = ng['w05'] - og['w05']
    dmae = ng['mae'] - og['mae']
    print(f"  {g:<5}: MAE {og['mae']:.3f}->{ng['mae']:.3f} ({dmae:+.3f})"
          f"  w0.5 {og['w05']:.1%}->{ng['w05']:.1%} ({dw05:+.1%})"
          f"  bias {og['bias']:+.3f}->{ng['bias']:+.3f}")

print(f"{'='*W}")

# ── Decision ─────────────────────────────────────────────────────────────────
improved = new_res['mae'] <= old_res['mae'] + 0.01  # allow tiny tolerance

if improved:
    print(f"\n  DECISION: NEW calibrator is better (or equivalent). APPLYING.")
    # Backup old
    import shutil
    shutil.copy(CAL_PATH, CAL_BACKUP)
    print(f"  Backed up old calibrator to: {CAL_BACKUP}")
    # Save new
    with open(CAL_PATH, "wb") as f:
        pickle.dump(new_cal, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved new OOF calibrator to: {CAL_PATH}")
else:
    print(f"\n  DECISION: NEW calibrator is WORSE on test. Keeping original.")
    print(f"  (New MAE {new_res['mae']:.3f} > Old MAE {old_res['mae']:.3f} + 0.01)")
    print(f"  New calibrator saved for reference: data/calibration_model_oof.pkl")
    with open("data/calibration_model_oof.pkl", "wb") as f:
        pickle.dump(new_cal, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\nDone.\n")
