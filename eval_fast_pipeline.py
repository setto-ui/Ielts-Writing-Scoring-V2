"""
eval_fast_pipeline.py — Compare Baseline vs TriSpecialist with Calibrator on X_val
Runs in 0.5s using cached X_val.
"""
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from scorer.two_stage_model import TriSpecialistModel
from scorer.calibration import calibrate, round_to_ielts_band
from scorer.ml_model import IELTSModel

data = np.load("data/cache_features_31.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]

# 1. Train and save TriSpecialistModel
model = TriSpecialistModel(alpha_low=0.6, alpha_high=0.6)
model.train(X_train, y_train, [f"f{i}" for i in range(31)])
model.save("data/model_tri_specialist.pkl")

# 2. Baseline Model
base_model = IELTSModel.load("data/model_exp_6JB.pkl")

# 3. Predict raw
pred_base_raw = base_model.predict(X_val)["overall"]
pred_tri_raw = model.predict(X_val)["overall"]

# 4. With Calibrator
pred_base_cal = np.array([calibrate(r) for r in pred_base_raw])
pred_tri_cal = np.array([calibrate(r) for r in pred_tri_raw])

def print_comp(name, yp, yt):
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

print_comp("Baseline Model 6J-B + Calibrator", pred_base_cal, y_val)
print_comp("Tri-Specialist Model + Calibrator", pred_tri_cal, y_val)
