"""
cv_benchmark_two_stage.py — 5-Fold Cross Validation of Tri-Specialist Architecture vs Baseline
Evaluates on TRAIN set (886 samples) using 5-fold CV to prove statistical robustness.
"""
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

data = np.load("data/cache_features_31.npz")
X_train, y_train = data["X_train"], data["y_train"]

def round_band(val):
    return np.round(np.clip(val, 4.0, 9.0) * 2.0) / 2.0

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_base_folds, acc_base_folds = [], []
mae_tri_folds, acc_tri_folds = [], []
low_acc_base_folds, low_acc_tri_folds = [], []
high_acc_base_folds, high_acc_tri_folds = [], []

print("Running 5-Fold Cross-Validation on 886 Train samples...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_vl, y_vl = X_train[val_idx], y_train[val_idx]

    # Baseline
    base_reg = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, min_child_samples=3, num_leaves=15, verbosity=-1, random_state=42)
    base_reg.fit(X_tr, y_tr)
    pred_base = base_reg.predict(X_vl)
    yp_base = round_band(pred_base)
    
    mae_base_folds.append(mean_absolute_error(y_vl, yp_base))
    acc_base_folds.append(np.mean(np.abs(yp_base - y_vl) <= 0.5) * 100)

    # Tri-Specialist
    clf_is_low = lgb.LGBMClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, min_child_samples=5, class_weight='balanced', verbosity=-1, random_state=42)
    clf_is_low.fit(X_tr, (y_tr <= 5.5).astype(int))
    p_is_low = clf_is_low.predict_proba(X_vl)[:, 1]

    clf_is_high = lgb.LGBMClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, min_child_samples=5, class_weight='balanced', verbosity=-1, random_state=42)
    clf_is_high.fit(X_tr, (y_tr >= 7.0).astype(int))
    p_is_high = clf_is_high.predict_proba(X_vl)[:, 1]

    reg_l = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, min_child_samples=3, num_leaves=15, verbosity=-1, random_state=42)
    reg_l.fit(X_tr[y_tr <= 5.5], y_tr[y_tr <= 5.5])
    p_l = reg_l.predict(X_vl)

    reg_h = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, min_child_samples=3, num_leaves=15, verbosity=-1, random_state=42)
    reg_h.fit(X_tr[y_tr >= 7.0], y_tr[y_tr >= 7.0])
    p_h = reg_h.predict(X_vl)

    w_l = p_is_low * 0.6
    w_h = p_is_high * 0.6
    w_b = np.maximum(0.0, 1.0 - w_l - w_h)
    total_w = w_b + w_l + w_h
    blend = (w_b * pred_base + w_l * p_l + w_h * p_h) / total_w
    yp_tri = round_band(blend)

    mae_tri_folds.append(mean_absolute_error(y_vl, yp_tri))
    acc_tri_folds.append(np.mean(np.abs(yp_tri - y_vl) <= 0.5) * 100)

    # Low / High breakdown
    l_idx = y_vl <= 5.5
    h_idx = y_vl >= 7.5
    if np.sum(l_idx) > 0:
        low_acc_base_folds.append(np.mean(np.abs(yp_base[l_idx] - y_vl[l_idx]) <= 0.5) * 100)
        low_acc_tri_folds.append(np.mean(np.abs(yp_tri[l_idx] - y_vl[l_idx]) <= 0.5) * 100)
    if np.sum(h_idx) > 0:
        high_acc_base_folds.append(np.mean(np.abs(yp_base[h_idx] - y_vl[h_idx]) <= 0.5) * 100)
        high_acc_tri_folds.append(np.mean(np.abs(yp_tri[h_idx] - y_vl[h_idx]) <= 0.5) * 100)

print(f"\n5-Fold CV Results (N=886):")
print(f"  Baseline 6J-B:         MAE = {np.mean(mae_base_folds):.3f} +- {np.std(mae_base_folds):.3f} | +-0.5 Acc = {np.mean(acc_base_folds):.1f}%")
print(f"  Tri-Specialist (Ours): MAE = {np.mean(mae_tri_folds):.3f} +- {np.std(mae_tri_folds):.3f} | +-0.5 Acc = {np.mean(acc_tri_folds):.1f}%")
print(f"  Low Band Acc (<=5.5):  Baseline {np.mean(low_acc_base_folds):.1f}% -> Tri-Specialist {np.mean(low_acc_tri_folds):.1f}% (+{np.mean(low_acc_tri_folds) - np.mean(low_acc_base_folds):.1f}%)")
print(f"  High Band Acc (>=7.5): Baseline {np.mean(high_acc_base_folds):.1f}% -> Tri-Specialist {np.mean(high_acc_tri_folds):.1f}% (+{np.mean(high_acc_tri_folds) - np.mean(high_acc_base_folds):.1f}%)")
