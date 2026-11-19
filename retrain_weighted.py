"""
retrain_weighted.py - Retrain Model 6J-B with inverse-frequency sample weights
                      + Retrain calibrator on train+val with same weights.

Fixes the "regression to mean" problem where band 4/9 essays get predicted
as band 6/7 due to severe class imbalance in training data.

Strategy:
1. Sample weights = 1 / frequency(band)  -> rare bands weighted much higher
2. Retrain LightGBM with these weights
3. Retrain isotonic calibrator on val set predictions vs true bands
4. Evaluate before/after on val.csv

STRICT RULE: test.csv is NOT touched.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm

print("Loading models and pipeline...", flush=True)

from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import compute_stats_from_dataset, normalize_features
from scorer.ml_model import IELTSModel
from scorer.calibration import train_calibrator, round_to_ielts_band


def compute_metrics(true_bands, pred_bands):
    tb = np.array(true_bands)
    pb = np.array(pred_bands)
    errors = np.abs(pb - tb)
    return {
        "mae":   float(np.mean(errors)),
        "rmse":  float(np.sqrt(np.mean(errors**2))),
        "bias":  float(np.mean(pb - tb)),
        "acc05": float(np.mean(errors <= 0.5) * 100),
    }


def extract_dataset(data):
    """Extract features + labels from dataset list."""
    features_list, y_list = [], []
    for sample in tqdm(data, desc="Extracting features"):
        essay  = sample.get("essay", "")
        prompt = sample.get("prompt", "")
        band   = float(sample.get("band", 5.0))
        if not essay.strip() or not prompt.strip():
            continue
        raw  = extract_all_features(essay, prompt)
        norm = normalize_features(raw)
        arr  = features_to_array(norm, FEATURE_KEYS)
        features_list.append(arr)
        y_list.append(band)
    return np.vstack(features_list), np.array(y_list)


def main():
    sep = "=" * 70
    print(f"\n{sep}")
    print("  RETRAIN 6J-B WITH SAMPLE WEIGHTS + RECALIBRATE")
    print(sep)

    # ── Load data ────────────────────────────────────────────────────────────
    train_data = _load_dataset(Path("data/train.csv"))
    val_data   = _load_dataset(Path("data/val.csv"))
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # ── Step 0: Recompute feature stats on train set ──────────────────────────
    print("\n[1/5] Recomputing feature stats...")
    all_raw = []
    for s in tqdm(train_data, desc="Stats"):
        r = extract_all_features(s.get("essay",""), s.get("prompt",""))
        all_raw.append(r)
    stats = compute_stats_from_dataset(all_raw)
    with open("data/feature_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  feature_stats.json updated ({len(stats)} features)")

    # ── Step 1: Extract train features ───────────────────────────────────────
    print("\n[2/5] Extracting train features...")
    X_train, y_train = extract_dataset(train_data)
    print(f"  Train matrix: {X_train.shape}")

    # ── Step 2: Compute inverse-frequency sample weights ─────────────────────
    print("\n[3/5] Computing sample weights (inverse frequency)...")
    band_counts = Counter(y_train)
    total = len(y_train)
    print(f"\n  Band distribution + weights:")
    print(f"  {'Band':<6} {'Count':>6} {'Freq':>7}  {'Weight':>8}")
    print(f"  {'-'*35}")

    weights = []
    for b in y_train:
        count = band_counts[b]
        freq  = count / total
        w     = 1.0 / freq          # inverse frequency
        weights.append(w)

    # Print summary
    for band in sorted(band_counts):
        count  = band_counts[band]
        freq   = count / total
        w      = 1.0 / freq
        bar    = "#" * min(count // 2, 40)
        print(f"  {band:<6} {count:>6} {freq:>6.1%}  {w:>8.1f}x  {bar}")

    sample_weights = np.array(weights)
    # Normalize weights to sum to n_samples (keeps learning rate stable)
    sample_weights = sample_weights / sample_weights.mean()
    print(f"\n  Weight range: {sample_weights.min():.2f}x — {sample_weights.max():.2f}x")

    # ── Step 3: Retrain model with weights ───────────────────────────────────
    print("\n[4/5] Training weighted LightGBM model...")
    model = IELTSModel()
    model.train(
        X=X_train,
        y={"overall": y_train},
        feature_names=FEATURE_KEYS,
        objective="regression",
        sample_weight=sample_weights,
    )
    model.save(Path("data/model_exp_6JB_weighted.pkl"))
    print("  Saved: data/model_exp_6JB_weighted.pkl")

    # ── Step 4: Retrain calibrator on val set ────────────────────────────────
    print("\n[5/5] Retraining calibrator on val set...")
    X_val, y_val = extract_dataset(val_data)
    raw_preds_val = model.predict(X_val)["overall"].flatten()

    train_calibrator(
        raw_means=raw_preds_val.tolist(),
        true_bands=y_val.tolist(),
        save_path="data/calibration_model_weighted.pkl",
    )
    print("  Saved: data/calibration_model_weighted.pkl")

    # ── Evaluate: Before vs After on val set ─────────────────────────────────
    print(f"\n{sep}")
    print("  EVALUATION ON VAL SET (N=190)")
    print(sep)

    # --- Old model (no weights) ---
    old_model = IELTSModel.load("data/model_exp_6JB.pkl")
    old_raw   = old_model.predict(X_val)["overall"].flatten()

    # Load old calibrator
    old_cal_path = Path("data/calibration_model.pkl")
    if old_cal_path.exists():
        with open(old_cal_path, "rb") as f:
            old_cal = pickle.load(f)
        old_pred = [round_to_ielts_band(float(old_cal.predict([r])[0])) for r in old_raw]
    else:
        old_pred = [round_to_ielts_band(float(r)) for r in old_raw]

    # --- New model (weighted) ---
    new_raw = raw_preds_val
    new_cal_path = Path("data/calibration_model_weighted.pkl")
    with open(new_cal_path, "rb") as f:
        new_cal = pickle.load(f)
    new_pred = [round_to_ielts_band(float(new_cal.predict([r])[0])) for r in new_raw]

    m_old = compute_metrics(y_val.tolist(), old_pred)
    m_new = compute_metrics(y_val.tolist(), new_pred)

    # Band-level breakdown
    buckets = [
        ("Low (4.0-5.5)",    4.0, 5.5),
        ("Mid (6.0-7.0)",    6.0, 7.0),
        ("High (7.5-9.0)",   7.5, 9.0),
    ]

    print(f"\n  {'Metric':<24} {'Old (no weights)':>18} {'New (weighted)':>16} {'Delta':>8}")
    print(f"  {'-'*70}")
    print(f"  {'Overall MAE':<24} {m_old['mae']:>18.3f} {m_new['mae']:>16.3f} {m_new['mae']-m_old['mae']:>+8.3f}")
    print(f"  {'Overall RMSE':<24} {m_old['rmse']:>18.3f} {m_new['rmse']:>16.3f} {m_new['rmse']-m_old['rmse']:>+8.3f}")
    print(f"  {'Overall +-0.5 Acc':<24} {m_old['acc05']:>17.1f}% {m_new['acc05']:>15.1f}% {m_new['acc05']-m_old['acc05']:>+7.1f}%")
    print(f"  {'Overall Bias':<24} {m_old['bias']:>+18.3f} {m_new['bias']:>+16.3f}")

    print(f"\n  --- Band-level breakdown ---")
    for name, lo, hi in buckets:
        idx_old = [i for i, b in enumerate(y_val) if lo <= b <= hi]
        if not idx_old:
            continue
        sub_t = [y_val[i] for i in idx_old]
        sub_o = [old_pred[i] for i in idx_old]
        sub_n = [new_pred[i] for i in idx_old]
        mo = compute_metrics(sub_t, sub_o)
        mn = compute_metrics(sub_t, sub_n)
        print(f"\n  {name} (N={len(idx_old)})")
        print(f"    MAE:     Old={mo['mae']:.3f}  New={mn['mae']:.3f}  Delta={mn['mae']-mo['mae']:+.3f}")
        print(f"    +-0.5:   Old={mo['acc05']:.1f}%  New={mn['acc05']:.1f}%  Delta={mn['acc05']-mo['acc05']:+.1f}%")
        print(f"    Bias:    Old={mo['bias']:+.3f}  New={mn['bias']:+.3f}")

    print(f"\n{sep}")

    # ── Decision prompt ──────────────────────────────────────────────────────
    print("\nKet qua tren. Neu new model tot hon, chay:")
    print("  python apply_weighted_model.py")
    print("de thay the model_exp_6JB.pkl va calibration_model.pkl chinh thuc.")
    print(sep)


if __name__ == "__main__":
    main()
