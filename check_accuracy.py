"""
check_accuracy.py - Xem model chấm chuẩn hay không
So sanh diem that (trong test.csv) vs diem model du doan, hien tung bai.

Usage:
    python check_accuracy.py           # hien 20 bai ngau nhien tu test.csv
    python check_accuracy.py --n 50    # hien 50 bai
    python check_accuracy.py --all     # hien toan bo 191 bai
"""

import argparse
import random
import numpy as np

print("Loading models, please wait...", flush=True)

from evaluate import _load_dataset, extract_all_features, features_to_array, FEATURE_KEYS
from scorer.normalizer import normalize_features
from scorer.ml_model import IELTSModel
from scorer.scorer import round_to_ielts_band
from pathlib import Path


def predict_band(essay, prompt, model):
    raw  = extract_all_features(essay, prompt)
    norm = normalize_features(raw)
    arr  = features_to_array(norm, FEATURE_KEYS).reshape(1, -1)
    raw_pred = float(model.predict(arr)["overall"].flatten()[0])
    return round_to_ielts_band(raw_pred), int(raw.get("word_count", 0))


def diff_symbol(delta):
    if abs(delta) <= 0.5:
        return "OK "
    elif delta > 0:
        return "HI "  # model cho cao hon thuc te
    else:
        return "LO "  # model cho thap hon thuc te


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int, default=20, help="So bai can hien (default: 20)")
    parser.add_argument("--all", action="store_true",  help="Hien toan bo test set")
    parser.add_argument("--seed",type=int, default=42,  help="Random seed")
    args = parser.parse_args()

    data = _load_dataset(Path("data/test.csv"))
    model = IELTSModel.load("data/model_exp_6JB.pkl")

    random.seed(args.seed)
    samples = data if args.all else random.sample(data, min(args.n, len(data)))

    print(f"\nModel: 6J-B (31 features) | Test set: {len(data)} bai | Hien: {len(samples)} bai\n")

    sep = "=" * 65
    print(sep)
    print(f"  {'#':<4} {'True':>6} {'Pred':>6} {'Delta':>7}  {'Status':<5}  {'Words':>5}  Prompt (30 chars)")
    print(sep)

    n_ok = n_hi = n_lo = 0
    errors = []

    for i, s in enumerate(samples, 1):
        essay  = s.get("essay", "")
        prompt = s.get("prompt", "")
        true_b = float(s.get("band", 5.0))

        if not essay.strip() or not prompt.strip():
            continue

        pred_b, wc = predict_band(essay, prompt, model)
        delta = pred_b - true_b
        sym   = diff_symbol(delta)

        if sym == "OK ": n_ok += 1
        elif sym == "HI ": n_hi += 1
        else: n_lo += 1

        errors.append(abs(delta))
        prompt_short = prompt.replace("\n", " ")[:35]
        print(f"  {i:<4} {true_b:>6.1f} {pred_b:>6.1f} {delta:>+7.1f}  {sym}  {wc:>5}  {prompt_short}")

    print(sep)
    n_total = len(errors)
    mae  = np.mean(errors)
    acc  = sum(1 for e in errors if e <= 0.5) / n_total * 100

    print(f"\n  Tong so bai : {n_total}")
    print(f"  Dung (<=0.5): {n_ok} bai  ({acc:.1f}%)")
    print(f"  Cao qua     : {n_hi} bai  ({n_hi/n_total*100:.1f}%)")
    print(f"  Thap qua    : {n_lo} bai  ({n_lo/n_total*100:.1f}%)")
    print(f"  MAE         : {mae:.3f} bands")
    print(f"\n  OK  = chenh lech <= 0.5 band (dat yeu cau)")
    print(f"  HI  = model cho cao hon thuc te > 0.5")
    print(f"  LO  = model cho thap hon thuc te > 0.5")
    print(sep)


if __name__ == "__main__":
    main()
