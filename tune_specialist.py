"""
tune_specialist.py — Fine-tune Tri-Specialist Blending with expanded boundaries
"""
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

data = np.load("data/cache_features_31.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]

def round_band(val):
    val = np.clip(val, 4.0, 9.0)
    return np.round(val * 2.0) / 2.0

def eval_metrics(y_true, y_pred_raw):
    y_pred = round_band(y_pred_raw)
    mae = mean_absolute_error(y_true, y_pred)
    acc05 = np.mean(np.abs(y_pred - y_true) <= 0.5) * 100.0
    bias = np.mean(y_pred - y_true)
    
    groups = {}
    for gname, lo, hi in [("Low  (4.0-5.5)", 4.0, 5.5), ("Mid  (6.0-7.0)", 6.0, 7.0), ("High (7.5-9.0)", 7.5, 9.0)]:
        idx = np.where((y_true >= lo) & (y_true <= hi))[0]
        if len(idx) > 0:
            yt, yp = y_true[idx], y_pred[idx]
            g_mae = mean_absolute_error(yt, yp)
            g_acc05 = np.mean(np.abs(yp - yt) <= 0.5) * 100.0
            g_bias = np.mean(yp - yt)
            groups[gname] = (len(idx), g_mae, g_bias, g_acc05)
    return mae, bias, acc05, groups

def print_res(name, y_true, y_pred_raw):
    mae, bias, acc05, groups = eval_metrics(y_true, y_pred_raw)
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    print(f"  Overall MAE: {mae:.3f} | Bias: {bias:+.3f} | +-0.5 Acc: {acc05:.1f}%")
    for gname, (n, g_mae, g_bias, g_acc05) in groups.items():
        print(f"    {gname:<16} (N={n:>3}): MAE={g_mae:.3f} | Bias={g_bias:+.3f} | +-0.5={g_acc05:>5.1f}%")
    return mae, acc05

# 1. Base Regressor
base_reg = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                             min_child_samples=3, num_leaves=15, objective="regression",
                             verbosity=-1, random_state=42)
base_reg.fit(X_train, y_train)
pred_base = base_reg.predict(X_val)

# 2. Binary Classifiers with balanced class weights
clf_is_low = lgb.LGBMClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, min_child_samples=5, class_weight="balanced", verbosity=-1, random_state=42)
clf_is_low.fit(X_train, (y_train <= 5.5).astype(int))
p_is_low = clf_is_low.predict_proba(X_val)[:, 1]

clf_is_high = lgb.LGBMClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, min_child_samples=5, class_weight="balanced", verbosity=-1, random_state=42)
clf_is_high.fit(X_train, (y_train >= 7.5).astype(int))
p_is_high = clf_is_high.predict_proba(X_val)[:, 1]

# 3. Test different Specialist Regressor cutoff thresholds
for low_cutoff in [5.5, 6.0]:
    for high_cutoff in [7.0, 7.5]:
        reg_l = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, min_child_samples=3, num_leaves=15, verbosity=-1, random_state=42)
        reg_l.fit(X_train[y_train <= low_cutoff], y_train[y_train <= low_cutoff])
        p_l = reg_l.predict(X_val)

        reg_h = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, min_child_samples=3, num_leaves=15, verbosity=-1, random_state=42)
        reg_h.fit(X_train[y_train >= high_cutoff], y_train[y_train >= high_cutoff])
        p_h = reg_h.predict(X_val)

        # Test blend weight scaling (alpha)
        for alpha in [0.5, 0.7, 1.0]:
            w_l = p_is_low * alpha
            w_h = p_is_high * alpha
            w_b = np.maximum(0.0, 1.0 - w_l - w_h)
            total_w = w_b + w_l + w_h
            blend = (w_b * pred_base + w_l * p_l + w_h * p_h) / total_w
            print_res(f"LowCut<={low_cutoff} | HighCut>={high_cutoff} | Alpha={alpha}", y_val, blend)
