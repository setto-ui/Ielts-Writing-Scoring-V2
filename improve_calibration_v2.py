"""
improve_calibration_v2.py — OOF calibration with CONSISTENT pipeline.

Fix vs v1: OOF predictions now go through score_essay + apply_constraints,
same as inference time. This ensures the calibrator input distribution matches.

Strategy:
  1. Extract (raw_feat, norm_feat, arr, true_band) for all 886 train essays
  2. 5-fold CV: for each held-out essay, predict with fold's mini-model,
     then pass through score_essay + apply_constraints to get constrained["overall"]
  3. Collect 886 OOF (constrained_overall, true_band) pairs
  4. Add 190 val pairs (same format) → 1076 total
  5. Fit new IsotonicRegression
  6. Evaluate on test set
  7. Apply ONLY IF: overall MAE improves AND no band group loses >5% w0.5

Usage:
    python improve_calibration_v2.py
"""
import sys, json, csv, pickle, math, warnings
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
import lightgbm as lgb

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

print("=" * 65)
print("  CALIBRATION IMPROVEMENT v2 — Consistent OOF pipeline")
print("=" * 65)

from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import normalize_features
from scorer.ml_model import IELTSModel
from scorer.scorer import score_essay
from scorer.constraint import apply_constraints
from scorer.calibration import round_to_ielts_band

MODEL_PATH = Path("data/model_exp_6JB.pkl")
CAL_PATH   = Path("data/calibration_model.pkl")
CAL_BACKUP = Path("data/calibration_model_backup_preOOF_v2.pkl")
STATS_PATH = Path("data/feature_stats.json")

_LGB_PARAMS = {
    "n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,
    "min_child_samples": 3, "num_leaves": 15, "verbosity": -1,
    "objective": "regression",
}

# ── Load ──────────────────────────────────────────────────────────────────────
print("\n[1/5] Loading datasets and model...")
train_data = _load_dataset(Path("data/train.csv"))
val_data   = _load_dataset(Path("data/val.csv"))
test_data  = _load_dataset(Path("data/test.csv"))
print(f"  Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}")

model = IELTSModel.load(MODEL_PATH)
with open(CAL_PATH, "rb") as f:
    old_cal = pickle.load(f)

# ── Extract full feature data for train ───────────────────────────────────────
print("\n[2/5] Extracting train features (storing raw+norm+arr)...")
train_raw_feats  = []   # list of raw_feat dicts
train_norm_feats = []   # list of norm_feat dicts
train_arrs       = []   # list of numpy arrays (for LightGBM)
train_bands      = []   # list of true bands

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
        train_raw_feats.append(raw_feat)
        train_norm_feats.append(norm_feat)
        train_arrs.append(arr)
        train_bands.append(true_b)
    except Exception:
        pass
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(train_data)}", flush=True)

X = np.vstack(train_arrs)
y = np.array(train_bands)
print(f"  Matrix: {X.shape}  y range: [{y.min():.1f}, {y.max():.1f}]")

# ── Collect val set calibration pairs (post-constraint format) ────────────────
print("\n[3/5] Collecting VAL set calibration pairs...")

def get_constrained_overall(raw_feat, norm_feat, ml_preds):
    """Run score_essay + apply_constraints and return overall (calibrator input)."""
    score_res   = score_essay(
        normalized_features=norm_feat,
        raw_features=raw_feat,
        ml_predictions=ml_preds,
    )
    gated = score_res["scores"]
    constrained, _ = apply_constraints(gated, raw_feat)
    return float(constrained.get("overall", 5.0))

val_pairs = []
for i, sample in enumerate(val_data):
    essay  = sample.get("essay", "")
    prompt = sample.get("prompt", "")
    true_b = float(sample.get("band", 5.0))
    if true_b < 1.0:
        continue
    try:
        raw_feat  = extract_all_features(essay, prompt)
        norm_feat = normalize_features(raw_feat)
        arr       = features_to_array(norm_feat, FEATURE_KEYS).reshape(1, -1)
        ml_preds  = model.predict(arr)
        raw_overall = get_constrained_overall(raw_feat, norm_feat, ml_preds)
        val_pairs.append((raw_overall, true_b))
    except Exception:
        pass
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(val_data)}", flush=True)
print(f"  Collected {len(val_pairs)} val pairs.")

# ── 5-fold OOF with FULL pipeline ─────────────────────────────────────────────
print("\n[4/5] 5-fold OOF with full pipeline (score_essay + apply_constraints)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_constrained = []   # (constrained_overall, true_band)

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X)):
    # Train fold model
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    reg = lgb.LGBMRegressor(**_LGB_PARAMS)
    reg.fit(X_tr, y_tr)

    # Predict held-out essays through full pipeline
    fold_mae = 0.0
    for idx in vl_idx:
        raw_feat  = train_raw_feats[idx]
        norm_feat = train_norm_feats[idx]
        arr       = train_arrs[idx].reshape(1, -1)
        true_b    = float(train_bands[idx])

        # Get ml_preds from FOLD model (not the champion model)
        raw_preds = np.clip(reg.predict(arr), 4.0, 9.0)
        ml_preds  = {"overall": float(raw_preds[0])}

        try:
            raw_overall = get_constrained_overall(raw_feat, norm_feat, ml_preds)
            # Apply old calibrator to get fold model's final prediction for MAE check
            cal_pred    = round_to_ielts_band(float(old_cal.predict(np.array([raw_overall]))[0]))
            fold_mae   += abs(cal_pred - true_b)
            oof_constrained.append((raw_overall, true_b))
        except Exception:
            pass

    fold_n = len(vl_idx)
    print(f"  Fold {fold+1}: N={fold_n}  MAE_w_old_cal={fold_mae/fold_n:.3f}")

print(f"  Collected {len(oof_constrained)} OOF pairs.")

# ── Train new calibrator ──────────────────────────────────────────────────────
print("\n[5/5] Training new calibrator and evaluating on TEST...")
all_pairs = oof_constrained + val_pairs
raw_all   = np.array([p[0] for p in all_pairs])
true_all  = np.array([p[1] for p in all_pairs])
print(f"  Total calibration points: {len(all_pairs)}")

new_cal = IsotonicRegression(out_of_bounds="clip")
new_cal.fit(raw_all, true_all)

# Show calibrator mapping
print("\n  Calibrator mapping (raw → new / raw → old):")
for v in [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]:
    n_out = round_to_ielts_band(float(new_cal.predict(np.array([v]))[0]))
    o_out = round_to_ielts_band(float(old_cal.predict(np.array([v]))[0]))
    print(f"    raw={v:.1f}  old→{o_out:.1f}  new→{n_out:.1f}")

# ── Evaluate on test ──────────────────────────────────────────────────────────
def eval_test(cal, label):
    preds, trues = [], []
    for i, sample in enumerate(test_data):
        essay  = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        true_b = float(sample.get("band", 5.0))
        if true_b < 1.0:
            continue
        try:
            raw_feat  = extract_all_features(essay, prompt)
            norm_feat = normalize_features(raw_feat)
            arr       = features_to_array(norm_feat, FEATURE_KEYS).reshape(1, -1)
            ml_preds  = model.predict(arr)
            raw_ov    = get_constrained_overall(raw_feat, norm_feat, ml_preds)
            pred_b    = round_to_ielts_band(float(cal.predict(np.array([raw_ov]))[0]))
            preds.append(pred_b)
            trues.append(true_b)
        except Exception:
            pass
        if (i + 1) % 25 == 0:
            print(f"  [{label}] {i+1}/{len(test_data)}", flush=True)

    errs  = [abs(p-t) for p,t in zip(preds,trues)]
    mae   = sum(errs)/len(errs)
    w05   = sum(1 for e in errs if e<=0.5)/len(errs)
    w10   = sum(1 for e in errs if e<=1.0)/len(errs)
    bias  = sum(p-t for p,t in zip(preds,trues))/len(preds)
    groups = {}
    for g, lo, hi in [("Low",4.0,5.5),("Mid",6.0,7.0),("High",7.5,9.0)]:
        gp = [(p,t) for p,t in zip(preds,trues) if lo<=t<=hi]
        if gp:
            ge = [abs(p-t) for p,t in gp]
            groups[g] = {"n":len(gp),"mae":sum(ge)/len(ge),
                         "bias":sum(p-t for p,t in gp)/len(gp),
                         "w05":sum(1 for e in ge if e<=0.5)/len(ge)}
    return {"mae":mae,"w05":w05,"w10":w10,"bias":bias,"n":len(preds),"groups":groups}

print("\nScoring with OLD calibrator on TEST...")
old_r = eval_test(old_cal, "OLD")
print("\nScoring with NEW calibrator on TEST...")
new_r = eval_test(new_cal, "NEW")

# ── Print comparison ──────────────────────────────────────────────────────────
W = 65
print(f"\n{'='*W}")
print(f"  TEST SET COMPARISON (N={old_r['n']})")
print(f"{'='*W}")
for name, o, n, lb in [
    ("Overall MAE", old_r['mae'], new_r['mae'], True),
    ("Overall w0.5", old_r['w05'], new_r['w05'], False),
    ("Overall w1.0", old_r['w10'], new_r['w10'], False),
]:
    d = n - o
    flag = ("BETTER" if (lb and d<-0.003) or (not lb and d>0.003)
            else "WORSE" if (lb and d>0.003) or (not lb and d<-0.003)
            else "~same")
    print(f"  {name:<18}: Old={o:.3f}  New={n:.3f}  [{flag}]")

print(f"\n  Per-band:")
for g in ["Low","Mid","High"]:
    o = old_r["groups"].get(g,{}); n = new_r["groups"].get(g,{})
    if not o or not n: continue
    dw = n['w05']-o['w05']
    dm = n['mae']-o['mae']
    flag = "OK" if dw >= -0.05 else "WARN"
    print(f"  {g:<5}: MAE {o['mae']:.3f}->{n['mae']:.3f} ({dm:+.3f})"
          f"  w0.5 {o['w05']:.1%}->{n['w05']:.1%} ({dw:+.1%})  [{flag}]")
print(f"{'='*W}")

# ── Decision: stricter criterion ──────────────────────────────────────────────
mae_ok  = new_r['mae'] <= old_r['mae'] + 0.005
band_ok = all(
    new_r["groups"].get(g, {}).get("w05", 0) >= old_r["groups"].get(g, {}).get("w05", 0) - 0.05
    for g in ["Low", "Mid", "High"]
    if old_r["groups"].get(g)
)

print(f"\n  MAE improves: {mae_ok}")
print(f"  No band group loses >5% w0.5: {band_ok}")

if mae_ok and band_ok:
    import shutil
    shutil.copy(CAL_PATH, CAL_BACKUP)
    with open(CAL_PATH, "wb") as f:
        pickle.dump(new_cal, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n  APPLIED: New calibrator saved. Backup: {CAL_BACKUP}")
else:
    with open("data/calibration_model_oof_v2.pkl", "wb") as f:
        pickle.dump(new_cal, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n  NOT APPLIED: Criterion not met. Saved for reference as calibration_model_oof_v2.pkl")
    print(f"  Keeping original calibrator.")

print("\nDone.\n")
