"""
audit_subscores.py — Phase 2: Sub-score criterion audit

For each of the 4 IELTS criteria (TR, CC, LR, GRA), computes:
  - MAE, bias, within_0.5, within_1.0 on VAL and TEST sets
  - Per-band-group breakdown per criterion
  - Prints comparison table + saves data/subscores_audit.json

Does NOT modify any model. Read-only.

Usage:
    python audit_subscores.py
"""

import sys, json, csv, math, random
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("Loading models...", flush=True)

from evaluate import run_full_pipeline, FEATURE_KEYS
from scorer.ml_model import IELTSModel

MODEL_PATH = "data/model_exp_6JB.pkl"
VAL_CSV    = "data/val.csv"
TEST_CSV   = "data/test.csv"
OUT_JSON   = "data/subscores_audit.json"

CRITERIA = [
    ("task_response", "Task_Response",     "TR"),
    ("coherence",     "Coherence_Cohesion","CC"),
    ("lexical",       "Lexical_Resource",  "LR"),
    ("grammar",       "Range_Accuracy",    "GRA"),
]

GROUPS = [("Low", 4.0, 5.5), ("Mid", 6.0, 7.0), ("High", 7.5, 9.0)]

RANDOM_SEED = 42


# ── Bootstrap CI ─────────────────────────────────────────────────────────────
def bootstrap_ci(values, stat_fn, n_iter=1000, ci=0.95, seed=RANDOM_SEED):
    rng = random.Random(seed)
    n   = len(values)
    if n < 3:
        return (0.0, 1.0)
    boot = sorted(stat_fn([values[rng.randint(0, n-1)] for _ in range(n)])
                  for _ in range(n_iter))
    lo = int((1 - ci) / 2 * n_iter)
    hi = int((1 + ci) / 2 * n_iter)
    return boot[lo], boot[hi]


# ── Stats for a list of (true, pred) pairs ───────────────────────────────────
def stats(pairs):
    if not pairs:
        return {}
    errs   = [p - t for t, p in pairs]
    abs_e  = [abs(e) for e in errs]
    n      = len(pairs)
    mae    = sum(abs_e) / n
    bias   = sum(errs) / n
    w05    = sum(1 for e in abs_e if e <= 0.5) / n
    w10    = sum(1 for e in abs_e if e <= 1.0) / n
    ci05   = bootstrap_ci(abs_e, lambda v: sum(1 for e in v if e <= 0.5) / len(v))
    return {
        "n": n, "mae": round(mae, 3), "bias": round(bias, 3),
        "w05": round(w05, 3), "w10": round(w10, 3),
        "ci05": [round(ci05[0], 3), round(ci05[1], 3)],
    }


# ── Run pipeline on one CSV, collect true + predicted sub-scores ──────────────
def run_csv(csv_path, model, label):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"\n  [{label}] Scoring {len(csv_path)} — {len(rows)} essays...", flush=True)
    results = []
    skipped = 0
    for i, row in enumerate(rows):
        prompt = (row.get("Question") or row.get("prompt") or "").strip()
        essay  = (row.get("Essay")    or row.get("essay")  or "").strip()

        # True overall band
        try:
            true_overall = float(row.get("Overall") or row.get("overall_band") or 0)
        except:
            skipped += 1; continue
        if true_overall < 1.0:
            skipped += 1; continue

        # True sub-scores (may be missing in some datasets)
        true_subs = {}
        for key, col, _ in CRITERIA:
            v = row.get(col, "").strip()
            try:
                true_subs[key] = float(v)
            except:
                true_subs[key] = None

        # Pipeline prediction
        try:
            result   = run_full_pipeline(essay, prompt, model)
            pred_overall = result["overall_band"]
            pred_subs    = result["sub_scores"]   # dict: criterion -> float
        except Exception as e:
            skipped += 1; continue

        results.append({
            "true_overall": true_overall,
            "pred_overall": pred_overall,
            "true_subs":    true_subs,
            "pred_subs":    pred_subs,
        })
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(rows)}", flush=True)

    if skipped:
        print(f"  Skipped {skipped} rows.")
    return results


# ── Print formatted table ─────────────────────────────────────────────────────
def print_criterion_table(label, crit_stats_overall, crit_stats_by_group):
    W = 75
    print(f"\n{'='*W}")
    print(f"  {label}")
    print(f"{'='*W}")
    print(f"  {'Crit':<5}  {'N':>5}  {'MAE':>6}  {'Bias':>7}  {'w0.5':>7}  {'CI_w0.5':>16}  {'w1.0':>7}")
    print(f"  {'-'*68}")
    for key, col, short in CRITERIA:
        s = crit_stats_overall.get(key, {})
        if not s:
            print(f"  {short:<5}  (no true labels available)")
            continue
        ci = f"[{s['ci05'][0]:.2f},{s['ci05'][1]:.2f}]"
        print(f"  {short:<5}  {s['n']:>5}  {s['mae']:>6.3f}  {s['bias']:>+7.3f}"
              f"  {s['w05']:>6.1%}  {ci:>16}  {s['w10']:>6.1%}")
    print(f"{'='*W}")

    # Per-group breakdown
    print(f"\n  Per-band-group breakdown:")
    print(f"  {'Crit':<5}  {'Group':<6}  {'N':>4}  {'MAE':>6}  {'Bias':>7}  {'w0.5':>7}  {'w1.0':>7}")
    print(f"  {'-'*52}")
    for key, col, short in CRITERIA:
        group_data = crit_stats_by_group.get(key, {})
        for gname, _, _ in GROUPS:
            s = group_data.get(gname, {})
            if not s:
                continue
            print(f"  {short:<5}  {gname:<6}  {s['n']:>4}  {s['mae']:>6.3f}"
                  f"  {s['bias']:>+7.3f}  {s['w05']:>6.1%}  {s['w10']:>6.1%}")
        print()
    print(f"{'='*W}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    model = IELTSModel.load(MODEL_PATH)

    val_results  = run_csv(VAL_CSV,  model, "VAL")
    test_results = run_csv(TEST_CSV, model, "TEST")

    audit = {}
    for split_label, results in [("val", val_results), ("test", test_results)]:
        # Overall sub-score stats (across all bands)
        crit_overall = {}
        crit_by_group = {}

        for key, col, short in CRITERIA:
            # Collect (true, pred) pairs where true label exists
            pairs_all = [
                (r["true_subs"][key], r["pred_subs"].get(key, r["pred_overall"]))
                for r in results
                if r["true_subs"].get(key) is not None
            ]
            crit_overall[key] = stats(pairs_all)

            # Per-band-group (by true OVERALL band)
            group_pairs = {}
            for gname, lo, hi in GROUPS:
                group_pairs[gname] = [
                    (r["true_subs"][key], r["pred_subs"].get(key, r["pred_overall"]))
                    for r in results
                    if r["true_subs"].get(key) is not None
                    and lo <= r["true_overall"] <= hi
                ]
            crit_by_group[key] = {g: stats(p) for g, p in group_pairs.items()}

        audit[split_label] = {
            "overall": crit_overall,
            "by_group": crit_by_group,
        }

        print_criterion_table(
            f"{split_label.upper()} SET — Sub-score accuracy per criterion",
            crit_overall,
            crit_by_group,
        )

    # ── Cross-split comparison ─────────────────────────────────────────────
    W = 75
    print(f"\n{'='*W}")
    print(f"  SUMMARY: VAL vs TEST sub-score accuracy")
    print(f"{'='*W}")
    print(f"  {'Crit':<5}  {'Val_w0.5':>10}  {'Test_w0.5':>10}  {'Val_MAE':>9}  {'Test_MAE':>9}  {'Val_bias':>9}  {'Test_bias':>9}")
    print(f"  {'-'*70}")
    for key, col, short in CRITERIA:
        vs = audit["val"]["overall"].get(key, {})
        ts = audit["test"]["overall"].get(key, {})
        if not vs or not ts:
            print(f"  {short:<5}  (missing data)")
            continue
        gap_w05 = ts["w05"] - vs["w05"]
        gap_str = f"({gap_w05:+.1%})"
        print(f"  {short:<5}  {vs['w05']:>9.1%}  {ts['w05']:>9.1%}  "
              f"  {vs['mae']:>8.3f}  {ts['mae']:>8.3f}  "
              f"  {vs['bias']:>+8.3f}  {ts['bias']:>+8.3f}  {gap_str}")
    print(f"{'='*W}")
    print(f"\n  Note: If Test_w0.5 << Val_w0.5 for a criterion, that criterion is overfitting.")
    print(f"  Note: Bias > 0 = model over-predicts; Bias < 0 = under-predicts.\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"  Saved: {OUT_JSON}\n")


if __name__ == "__main__":
    main()
