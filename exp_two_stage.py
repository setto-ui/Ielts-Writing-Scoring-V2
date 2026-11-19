"""
exp_two_stage.py — Fast Benchmark of Two-Stage and Specialist Architectures
Uses pre-computed data/cache_features_31.npz for instant execution (< 2 seconds).
Evaluates strictly on Validation Set (N=190).
"""
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# 1. Load cached features
data = np.load("data/cache_features_31.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]

print(f"Loaded: Train {X_train.shape}, Val {X_val.shape}")

def round_band(val):
    val = np.clip(val, 4.0, 9.0)
    return np.round(val * 2.0) / 2.0

def eval_metrics(y_true, y_pred_raw):
    y_pred = round_band(y_pred_raw)
    mae = mean_absolute_error(y_true, y_pred)
    acc05 = np.mean(np.abs(y_pred - y_true) <= 0.5) * 100.0
    bias = np.mean(y_pred - y_true)
    
    # Per group
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

def print_result(name, y_true, y_pred_raw):
    mae, bias, acc05, groups = eval_metrics(y_true, y_pred_raw)
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    print(f"  Overall MAE: {mae:.3f} | Bias: {bias:+.3f} | +-0.5 Acc: {acc05:.1f}%")
    print(f"  Per-group:")
    for gname, (n, g_mae, g_bias, g_acc05) in groups.items():
        print(f"    {gname:<16} (N={n:>3}): MAE={g_mae:.3f} | Bias={g_bias:+.3f} | +-0.5={g_acc05:>5.1f}%")
    return mae, acc05

# -------------------------------------------------------------
# Baseline: Single LightGBM Regressor (Model 6J-B style)
# -------------------------------------------------------------
base_reg = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                             min_child_samples=3, num_leaves=15, objective="regression",
                             verbosity=-1, random_state=42)
base_reg.fit(X_train, y_train)
pred_base = base_reg.predict(X_val)
print_result("Baseline Single Regressor (31 features, L2)", y_val, pred_base)

# -------------------------------------------------------------
# Architecture 1: Phase 6F Soft/Hard Routing (Unweighted vs Balanced)
# -------------------------------------------------------------
def band_to_group(b):
    if b <= 5.5: return 0
    elif b <= 7.0: return 1
    else: return 2

y_tr_grp = np.array([band_to_group(b) for b in y_train])

# Regressors per group
reg_low = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, min_child_samples=3, num_leaves=15, verbosity=-1, random_state=42)
reg_mid = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, min_child_samples=3, num_leaves=15, verbosity=-1, random_state=42)
reg_high = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, min_child_samples=3, num_leaves=15, verbosity=-1, random_state=42)

reg_low.fit(X_train[y_tr_grp == 0], y_train[y_tr_grp == 0])
reg_mid.fit(X_train[y_tr_grp == 1], y_train[y_tr_grp == 1])
reg_high.fit(X_train[y_tr_grp == 2], y_train[y_tr_grp == 2])

p_low = reg_low.predict(X_val)
p_mid = reg_mid.predict(X_val)
p_high = reg_high.predict(X_val)

# Classifier 1a: Unweighted multiclass (Phase 6F exact)
clf_unweighted = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, min_child_samples=3, num_leaves=15, objective="multiclass", num_class=3, verbosity=-1, random_state=42)
clf_unweighted.fit(X_train, y_tr_grp)
probs_unw = clf_unweighted.predict_proba(X_val)
pred_soft_unw = probs_unw[:, 0]*p_low + probs_unw[:, 1]*p_mid + probs_unw[:, 2]*p_high
print_result("Architecture 1A: 6F Soft Routing (Unweighted Classifier)", y_val, pred_soft_unw)

# Classifier 1b: Balanced multiclass
clf_balanced = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, min_child_samples=3, num_leaves=15, objective="multiclass", num_class=3, class_weight="balanced", verbosity=-1, random_state=42)
clf_balanced.fit(X_train, y_tr_grp)
probs_bal = clf_balanced.predict_proba(X_val)
pred_soft_bal = probs_bal[:, 0]*p_low + probs_bal[:, 1]*p_mid + probs_bal[:, 2]*p_high
print_result("Architecture 1B: 6F Soft Routing (Balanced Classifier)", y_val, pred_soft_bal)

# -------------------------------------------------------------
# Architecture 2: Ordinal Detection + Tail Boosting (Frank-Hall)
# -------------------------------------------------------------
# Train binary detector: Is this a Low band essay (<= 5.5)?
y_tr_is_low = (y_train <= 5.5).astype(int)
clf_is_low = lgb.LGBMClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, min_child_samples=5, class_weight="balanced", verbosity=-1, random_state=42)
clf_is_low.fit(X_train, y_tr_is_low)
p_is_low = clf_is_low.predict_proba(X_val)[:, 1]

# Train binary detector: Is this a High band essay (>= 7.5)?
y_tr_is_high = (y_train >= 7.5).astype(int)
clf_is_high = lgb.LGBMClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, min_child_samples=5, class_weight="balanced", verbosity=-1, random_state=42)
clf_is_high.fit(X_train, y_tr_is_high)
p_is_high = clf_is_high.predict_proba(X_val)[:, 1]

# Adjust baseline prediction using tail probabilities
# If high probability of being High band -> nudge upward; if high probability of being Low band -> nudge downward
for high_weight, low_weight in [(0.5, 0.5), (0.7, 0.7), (1.0, 1.0)]:
    pred_ordinal_adj = pred_base.copy()
    # Continuous tail push
    pred_ordinal_adj = pred_ordinal_adj + (p_is_high * high_weight) - (p_is_low * low_weight)
    print_result(f"Architecture 2: Ordinal Tail Push (w_high={high_weight}, w_low={low_weight})", y_val, pred_ordinal_adj)

# -------------------------------------------------------------
# Architecture 3: Specialist Blending (Main Regressor + Specialist Extrapolator)
# -------------------------------------------------------------
# Instead of separate isolated regressors that can't see the rest of the data,
# blend Main regressor with High Specialist and Low Specialist
pred_specialist_blend = (1.0 - p_is_low - p_is_high) * pred_base + p_is_low * p_low + p_is_high * p_high
print_result("Architecture 3: Tri-Specialist Probability Blending", y_val, pred_specialist_blend)
