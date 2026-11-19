"""
debug_pipeline.py - Diagnose pipeline at each stage for a sample of val essays.
Shows raw model output, post-constraint, post-calibration for each essay.
"""
import sys, json, csv
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("Loading...", flush=True)

from evaluate import FEATURE_KEYS, extract_all_features, features_to_array
from scorer.normalizer import normalize_features
from scorer.ml_model import IELTSModel
from scorer.scorer import score_essay, round_to_ielts_band
from scorer.constraint import apply_constraints
from scorer.calibration import calibrate, load_calibrator
import numpy as np

MODEL_PATH = "data/model_exp_6JB.pkl"
model = IELTSModel.load(MODEL_PATH)
with open("data/feature_stats.json", encoding="utf-8") as f:
    feat_stats = json.load(f)

# Try to load calibrator directly and inspect it
try:
    cal = load_calibrator()
    print(f"\nCalibrator type  : {type(cal)}")
    if hasattr(cal, 'X_thresholds_'):
        xs = cal.X_thresholds_
        ys = cal.y_thresholds_
        print(f"Calibrator X range: {xs[0]:.2f} → {xs[-1]:.2f}")
        print(f"Calibrator Y range: {ys[0]:.2f} → {ys[-1]:.2f}")
        print(f"Calibrator mapping (X→Y):")
        for x, y in zip(xs, ys):
            print(f"  {x:.2f} → {y:.2f}")
    elif hasattr(cal, 'predict'):
        test_inputs = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
        print("Calibrator predict:")
        for v in test_inputs:
            try:
                out = cal.predict([[v]])[0]
                print(f"  {v:.1f} → {out:.2f}")
            except Exception as e:
                print(f"  {v:.1f} → ERROR: {e}")
except Exception as e:
    print(f"Could not load calibrator: {e}")

# Score 5 essays from each band group
print("\n" + "="*70)
print("  PIPELINE DEBUG — 5 essays per band group")
print("="*70)

with open("data/val.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Collect up to 5 per group
groups = {"Low": [], "Mid": [], "High": []}
for row in rows:
    try:
        b = float(row.get("Overall", 0))
    except:
        continue
    if 4.0 <= b <= 5.5 and len(groups["Low"]) < 5:
        groups["Low"].append(row)
    elif 6.0 <= b <= 7.0 and len(groups["Mid"]) < 5:
        groups["Mid"].append(row)
    elif 7.5 <= b <= 9.0 and len(groups["High"]) < 5:
        groups["High"].append(row)

for g, g_rows in groups.items():
    print(f"\n  --- {g} band ---")
    print(f"  {'True':>5}  {'Raw_overall':>12}  {'Post_constr':>12}  {'Post_calib':>12}  {'Final':>7}")
    print(f"  {'-'*55}")
    for row in g_rows:
        prompt = row.get("Question", "")
        essay  = row.get("Essay", "")
        true_b = float(row.get("Overall", 0))
        try:
            feats = extract_all_features(essay, prompt)
            arr   = features_to_array(feats, FEATURE_KEYS)
            norm  = normalize_features(arr, feat_stats)
            raw   = model.predict(norm)          # dict of criteria
            raw_overall = raw.get("overall", -1)

            sub_c = score_essay(raw)             # after score_essay
            sub_c2 = apply_constraints(sub_c.copy(), essay)  # after constraints
            sub_c2_overall = sub_c2.get("overall", -1)

            sub_cal = calibrate(sub_c2.copy())   # after calibration
            sub_cal_overall = sub_cal.get("overall", -1)

            final = round_to_ielts_band(sub_cal_overall)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print(f"  {true_b:>5.1f}  {raw_overall:>12.3f}  {sub_c2_overall:>12.3f}  {sub_cal_overall:>12.3f}  {final:>7.1f}")

print("\nDone.")
