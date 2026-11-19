"""
apply_weighted_model.py - Apply weighted model as the official model.
Run this ONLY after reviewing retrain_weighted.py results and deciding to adopt.
"""
import shutil
from pathlib import Path

print("Applying weighted model as official...")

src_model = Path("data/model_exp_6JB_weighted.pkl")
src_cal   = Path("data/calibration_model_weighted.pkl")
dst_model = Path("data/model_exp_6JB.pkl")
dst_cal   = Path("data/calibration_model.pkl")

# Backup originals
shutil.copy(dst_model, "data/model_exp_6JB_original.pkl")
shutil.copy(dst_cal,   "data/calibration_model_original.pkl")
print("  Backup saved: model_exp_6JB_original.pkl, calibration_model_original.pkl")

# Apply
shutil.copy(src_model, dst_model)
shutil.copy(src_cal,   dst_cal)
print("  Applied: model_exp_6JB.pkl <- weighted version")
print("  Applied: calibration_model.pkl <- weighted calibrator")
print("\nDone! Run test_scorer.py to verify.")
