"""
apply_mae_model.py — Apply the MAE model as champion + retrain calibrator to match.

Steps:
  1. Backup current MSE model
  2. Copy MAE model as model_exp_6JB.pkl
  3. Retrain calibrator using MAE model predictions on val set
  4. Final evaluation on test set
  5. Update confidence numbers

Usage:
    python apply_mae_model.py
"""
import sys, json, pickle, shutil, warnings
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

print("=" * 65)
print("  APPLYING MAE MODEL + RECALIBRATING")
print("=" * 65)

from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import normalize_features
from scorer.ml_model import IELTSModel
from scorer.scorer import score_essay
from scorer.constraint import apply_constraints
from scorer.calibration import train_calibrator, calibrate, round_to_ielts_band

MODEL_PATH     = Path("data/model_exp_6JB.pkl")
MODEL_MAE_PATH = Path("data/model_exp_6JB_mae.pkl")
MODEL_MSE_BAK  = Path("data/model_exp_6JB_mse_backup.pkl")
CAL_PATH       = Path("data/calibration_model.pkl")
CAL_BAK        = Path("data/calibration_model_mse_backup.pkl")

# ── Step 1: Backup + apply MAE model ──────────────────────────────────────────
print("\n[1/4] Backing up MSE model and applying MAE model...")
shutil.copy(MODEL_PATH, MODEL_MSE_BAK)
shutil.copy(MODEL_MAE_PATH, MODEL_PATH)
print(f"  MSE backup: {MODEL_MSE_BAK}")
print(f"  New champion: {MODEL_PATH} (MAE objective)")

model = IELTSModel.load(MODEL_PATH)

# ── Step 2: Retrain calibrator on val predictions from MAE model ───────────────
print("\n[2/4] Retraining calibrator using MAE model + val set...")
val_data = _load_dataset(Path("data/val.csv"))

def get_constrained_overall(essay, prompt, model):
    raw_feat  = extract_all_features(essay, prompt)
    norm_feat = normalize_features(raw_feat)
    arr       = features_to_array(norm_feat, FEATURE_KEYS).reshape(1, -1)
    ml_preds  = model.predict(arr)
    score_res = score_essay(
        normalized_features=norm_feat,
        raw_features=raw_feat,
        ml_predictions=ml_preds,
    )
    gated = score_res["scores"]
    constrained, _ = apply_constraints(gated, raw_feat)
    return float(constrained.get("overall", 5.0))

raw_preds, true_bands = [], []
for i, sample in enumerate(val_data):
    essay  = sample.get("essay", "")
    prompt = sample.get("prompt", "")
    true_b = float(sample.get("band", 5.0))
    if true_b < 1.0 or not essay.strip():
        continue
    try:
        raw_ov = get_constrained_overall(essay, prompt, model)
        raw_preds.append(raw_ov)
        true_bands.append(true_b)
    except Exception:
        pass
    if (i+1) % 25 == 0:
        print(f"  {i+1}/{len(val_data)}", flush=True)

shutil.copy(CAL_PATH, CAL_BAK)
new_cal = train_calibrator(raw_preds, true_bands, save_path=str(CAL_PATH))
print(f"  Calibrator retrained on {len(raw_preds)} val pairs.")
print(f"  Old calibrator backup: {CAL_BAK}")

# Show new calibrator mapping
print("\n  Calibrator mapping (raw_overall -> calibrated):")
for v in [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]:
    out = round_to_ielts_band(float(new_cal.predict(np.array([v]))[0]))
    print(f"    {v:.1f} -> {out:.1f}")

# ── Step 3: Full evaluation on test set ───────────────────────────────────────
print("\n[3/4] Final evaluation on TEST set (locked)...")
test_data = _load_dataset(Path("data/test.csv"))

def eval_full(dataset, label):
    preds, trues = [], []
    for i, sample in enumerate(dataset):
        essay  = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        true_b = float(sample.get("band", 5.0))
        if true_b < 1.0 or not essay.strip():
            continue
        try:
            raw_ov = get_constrained_overall(essay, prompt, model)
            pred_b = round_to_ielts_band(float(new_cal.predict(np.array([raw_ov]))[0]))
            preds.append(pred_b)
            trues.append(true_b)
        except Exception:
            pass
        if (i+1) % 25 == 0:
            print(f"  [{label}] {i+1}/{len(dataset)}", flush=True)
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
            groups[g] = {
                "n":len(gp), "mae":round(sum(ge)/len(ge),3),
                "bias":round(sum(p-t for p,t in gp)/len(gp),3),
                "w05":round(sum(1 for e in ge if e<=0.5)/len(ge),3),
                "w10":round(sum(1 for e in ge if e<=1.0)/len(ge),3),
            }
    return {"mae":round(mae,3),"w05":round(w05,3),"w10":round(w10,3),
            "bias":round(bias,3),"n":len(preds),"groups":groups}

result = eval_full(test_data, "TEST")

# ── Step 4: Print final summary ───────────────────────────────────────────────
W = 65
print(f"\n{'='*W}")
print(f"  FINAL MODEL STATS — MAE objective + recalibrated")
print(f"  Test set N={result['n']}")
print(f"{'='*W}")
print(f"  Overall MAE  : {result['mae']:.3f}")
print(f"  Overall w0.5 : {result['w05']:.1%}")
print(f"  Overall w1.0 : {result['w10']:.1%}")
print(f"  Overall Bias : {result['bias']:+.3f}")
print(f"\n  Per-band-group:")
print(f"  {'Group':<6}  {'N':>4}  {'MAE':>6}  {'Bias':>7}  {'w0.5':>7}  {'w1.0':>7}")
print(f"  {'-'*46}")
for g in ["Low","Mid","High"]:
    s = result["groups"].get(g,{})
    if not s: continue
    print(f"  {g:<6}  {s['n']:>4}  {s['mae']:>6.3f}  {s['bias']:>+7.3f}"
          f"  {s['w05']:>6.1%}  {s['w10']:>6.1%}")
print(f"{'='*W}")

# Save final audit to JSON for reference
audit = {"model": "6JB-MAE", "objective": "regression_l1",
         "calibrator": "IsotonicRegression-val-190", "test_results": result}
with open("data/final_model_audit.json", "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)

print(f"\n  Saved: data/final_model_audit.json")
print(f"\n  CONFIDENCE NUMBERS TO USE IN test_scorer.py:")
for g in ["Low","Mid","High"]:
    s = result["groups"].get(g,{})
    if not s: continue
    pct = int(round(s['w05']*100))
    mae = s['mae']
    spread = round(mae * 1.5 * 2) / 2
    spread = max(0.5, min(2.0, spread))
    label = "HIGH" if pct >= 65 else ("MEDIUM" if pct >= 45 else "LOW")
    print(f"    {g}: w0.5={pct}%  label={label}  spread=+-{spread:.1f}")

print("\nDone.\n")
