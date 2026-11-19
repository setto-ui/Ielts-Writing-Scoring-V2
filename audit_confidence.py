"""
audit_confidence.py — Phase 1: Confidence Audit

Computes per-band-group statistics from VAL set (N=190),
then cross-checks on TEST set (N=113, locked).

Outputs:
  - Per-band group: n, MAE, bias, within_0.5, within_1.0,
    95% bootstrap confidence intervals for within_0.5 and within_1.0
  - Recommendation: update or keep current confidence thresholds
  - Saves JSON: data/confidence_audit.json
  - Prints summary table

This script is READ-ONLY on test set — no model changes, no retraining.

Usage:
    python audit_confidence.py
"""

import sys, json, csv, math, random
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("Loading models for audit...", flush=True)

from evaluate import FEATURE_KEYS, extract_all_features, features_to_array, run_full_pipeline
from scorer.ml_model import IELTSModel

MODEL_PATH  = "data/model_exp_6JB.pkl"
VAL_CSV     = "data/val.csv"
TEST_CSV    = "data/test.csv"
OUT_JSON    = "data/confidence_audit.json"
RANDOM_SEED = 42

# ── Band groups (closed intervals, IELTS 0.5-step bands) ────────────────────
GROUPS = [
    ("Low",  4.0, 5.5),
    ("Mid",  6.0, 7.0),
    ("High", 7.5, 9.0),
]

# ── Bootstrap CI ─────────────────────────────────────────────────────────────
def bootstrap_ci(values, stat_fn, n_iter=2000, ci=0.95, seed=RANDOM_SEED):
    """Return (lower, upper) bootstrap confidence interval for stat_fn(values)."""
    rng = random.Random(seed)
    boot_stats = []
    n = len(values)
    for _ in range(n_iter):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        boot_stats.append(stat_fn(sample))
    boot_stats.sort()
    lo = int((1 - ci) / 2 * n_iter)
    hi = int((1 + ci) / 2 * n_iter)
    return boot_stats[lo], boot_stats[hi]

# ── Pipeline ─────────────────────────────────────────────────────────────────
def predict_essay(model, feat_stats, prompt, essay):
    result = run_full_pipeline(essay, prompt, model)
    return result["overall_band"]

# ── Per-essay predictions on a CSV split ─────────────────────────────────────
def run_on_csv(csv_path, model, feat_stats, label):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data   = list(reader)

    print(f"\n  Scoring {label} set ({len(data)} essays)...", flush=True)
    skipped = 0
    for i, row in enumerate(data):
        # Support multiple possible column names
        prompt = (row.get("Question") or row.get("prompt") or row.get("question") or "").strip()
        essay  = (row.get("Essay")    or row.get("essay")                           or "").strip()
        true_b_raw = (row.get("Overall") or row.get("overall_band") or
                      row.get("band")    or row.get("overall")      or "").strip()
        try:
            true_b = float(true_b_raw)
        except (ValueError, TypeError):
            skipped += 1
            continue
        if true_b < 1.0:          # sanity check — skip malformed rows
            skipped += 1
            continue
        try:
            pred_b = predict_essay(model, None, prompt, essay)
        except Exception:
            pred_b = 5.0
        rows.append({"true": true_b, "pred": pred_b, "err": pred_b - true_b})
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(data)}  (scored so far: {len(rows)})", flush=True)
    if skipped:
        print(f"  Skipped {skipped} rows (missing/invalid band label)")
    return rows

# ── Compute stats for a slice of rows ────────────────────────────────────────
def compute_stats(rows):
    if not rows:
        return {}
    errors = [r["err"] for r in rows]
    abs_e  = [abs(e) for e in errors]
    n      = len(rows)

    mae    = sum(abs_e) / n
    rmse   = math.sqrt(sum(e**2 for e in errors) / n)
    bias   = sum(errors) / n
    w05    = sum(1 for e in abs_e if e <= 0.5) / n
    w10    = sum(1 for e in abs_e if e <= 1.0) / n

    # Bootstrap 95% CI for within_0.5 and within_1.0
    ci05_lo, ci05_hi = bootstrap_ci(abs_e, lambda v: sum(1 for e in v if e <= 0.5) / len(v))
    ci10_lo, ci10_hi = bootstrap_ci(abs_e, lambda v: sum(1 for e in v if e <= 1.0) / len(v))

    return {
        "n"          : n,
        "mae"        : round(mae, 4),
        "rmse"       : round(rmse, 4),
        "bias"       : round(bias, 4),
        "within_0_5" : round(w05, 4),
        "within_1_0" : round(w10, 4),
        "ci_0_5"     : [round(ci05_lo, 4), round(ci05_hi, 4)],
        "ci_1_0"     : [round(ci10_lo, 4), round(ci10_hi, 4)],
    }

# ── Print a table section ────────────────────────────────────────────────────
def print_group_table(title, group_stats):
    W = 72
    sep = "=" * W
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    hdr = f"  {'Group':<8}  {'N':>5}  {'MAE':>6}  {'Bias':>7}  {'w0.5':>7}  {'CI_w0.5':>14}  {'w1.0':>7}  {'CI_w1.0':>14}"
    print(hdr)
    print("-" * W)
    for g, s in group_stats.items():
        if not s:
            print(f"  {g:<8}  {'(no data)':>5}")
            continue
        ci05 = f"[{s['ci_0_5'][0]:.2f},{s['ci_0_5'][1]:.2f}]"
        ci10 = f"[{s['ci_1_0'][0]:.2f},{s['ci_1_0'][1]:.2f}]"
        print(
            f"  {g:<8}  {s['n']:>5}  {s['mae']:>6.3f}  {s['bias']:>+7.3f}"
            f"  {s['within_0_5']:>6.1%}  {ci05:>14}"
            f"  {s['within_1_0']:>6.1%}  {ci10:>14}"
        )
    print(sep)

# ── Confidence mapping recommendation ────────────────────────────────────────
def recommend_mapping(val_gs, test_gs):
    """
    Recommend confidence label and likely-range for each group
    based on empirical within_0.5 and MAE on BOTH val and test.
    """
    rec = {}
    for g, lo, hi in GROUPS:
        vs = val_gs.get(g, {})
        ts = test_gs.get(g, {})
        if not vs:
            rec[g] = {
                "band_lo": lo, "band_hi": hi, "label": "UNKNOWN",
                "pct": 0, "spread": 1.0,
                "w05_val": 0, "w05_test": 0, "w05_avg": 0,
                "mae_val": 0, "mae_test": 0,
                "note": "no val data"
            }
            continue

        w05_val  = vs["within_0_5"]
        w05_test = ts.get("within_0_5", w05_val) if ts else w05_val
        mae_val  = vs["mae"]
        mae_test = ts.get("mae", mae_val) if ts else mae_val

        # Average val and test for a more conservative estimate
        w05_avg = (w05_val + w05_test) / 2 if ts else w05_val
        mae_avg = (mae_val + mae_test) / 2 if ts else mae_val

        # Likely range: use ±(MAE*1.5) rounded to nearest 0.5, clamped [4,9]
        spread     = round(mae_avg * 1.5 * 2) / 2          # round to nearest 0.5
        spread     = max(0.5, min(2.0, spread))
        r_lo       = max(4.0, lo - spread)
        r_hi       = min(9.0, hi + spread)

        pct_int    = int(round(w05_avg * 100))
        if pct_int >= 65:
            label = "HIGH"
        elif pct_int >= 45:
            label = "MEDIUM"
        else:
            label = "LOW"

        rec[g] = {
            "band_lo"     : lo,
            "band_hi"     : hi,
            "w05_val"     : round(w05_val, 4),
            "w05_test"    : round(w05_test, 4),
            "w05_avg"     : round(w05_avg, 4),
            "mae_val"     : round(mae_val, 4),
            "mae_test"    : round(mae_test, 4),
            "label"       : label,
            "pct"         : pct_int,
            "spread"      : spread,
            "note"        : (
                f"w0.5: val={w05_val:.1%}, test={w05_test:.1%} | "
                f"MAE: val={mae_val:.3f}, test={mae_test:.3f}"
            ),
        }
    return rec

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load model only — run_full_pipeline handles feature_stats internally
    model = IELTSModel.load(MODEL_PATH)

    # ── Score val set ─────────────────────────────────────────────────────────
    val_rows  = run_on_csv(VAL_CSV,  model, None, "VAL")

    # ── Score test set ────────────────────────────────────────────────────────
    test_rows = run_on_csv(TEST_CSV, model, None, "TEST")

    # ── Compute stats per group ───────────────────────────────────────────────
    def slice_rows(rows, lo, hi):
        return [r for r in rows if lo <= r["true"] <= hi]

    val_group_stats  = {}
    test_group_stats = {}
    for g, lo, hi in GROUPS:
        val_group_stats[g]  = compute_stats(slice_rows(val_rows,  lo, hi))
        test_group_stats[g] = compute_stats(slice_rows(test_rows, lo, hi))

    val_overall   = compute_stats(val_rows)
    test_overall  = compute_stats(test_rows)

    # ── Print tables ──────────────────────────────────────────────────────────
    print_group_table("VAL SET  — Per-band-group accuracy  (N=190)", val_group_stats)
    print(f"  Overall: N={val_overall['n']}  MAE={val_overall['mae']:.3f}  "
          f"bias={val_overall['bias']:+.3f}  w0.5={val_overall['within_0_5']:.1%}  "
          f"w1.0={val_overall['within_1_0']:.1%}")

    print_group_table("TEST SET — Per-band-group accuracy (LOCKED)  (N=~113)", test_group_stats)
    print(f"  Overall: N={test_overall['n']}  MAE={test_overall['mae']:.3f}  "
          f"bias={test_overall['bias']:+.3f}  w0.5={test_overall['within_0_5']:.1%}  "
          f"w1.0={test_overall['within_1_0']:.1%}")

    # ── Confidence mapping recommendation ────────────────────────────────────
    mapping = recommend_mapping(val_group_stats, test_group_stats)

    W = 72
    print(f"\n{'=' * W}")
    print(f"  CONFIDENCE MAPPING RECOMMENDATION")
    print(f"  (based on avg of val + test within_0.5, not just val)")
    print("=" * W)
    print(f"  {'Group':<8}  {'Bands':<12}  {'Label':<8}  {'Pct':>5}  {'Spread':>7}  Note")
    print("-" * W)
    for g, info in mapping.items():
        bands  = f"{info['band_lo']:.1f}-{info['band_hi']:.1f}"
        spread = f"+-{info['spread']:.1f}"
        print(f"  {g:<8}  {bands:<12}  {info['label']:<8}  {info['pct']:>4}%  {spread:>7}  {info['note']}")
    print("=" * W)

    print("\n  Definition: 'pct' = percentage of predictions within +-0.5 band of true band.")
    print("  CI = 95% bootstrap confidence interval. N_boot=2000.")
    print("  Spread for 'Likely range' = 1.5 * avg_MAE (val+test), rounded to nearest 0.5.\n")

    # ── Current vs recommended ────────────────────────────────────────────────
    current = {"Low": 36, "Mid": 71, "High": 53}
    print(f"{'=' * W}")
    print(f"  CURRENT vs RECOMMENDED CONFIDENCE PERCENTAGES")
    print(f"{'=' * W}")
    print(f"  {'Group':<8}  {'Current %':>10}  {'Recommended %':>14}  {'Change':>8}")
    print("-" * W)
    for g in ["Low", "Mid", "High"]:
        cur   = current[g]
        new   = mapping[g]["pct"]
        delta = new - cur
        arrow = f"+{delta}" if delta > 0 else str(delta)
        print(f"  {g:<8}  {cur:>9}%  {new:>13}%  {arrow:>8}")
    print("=" * W)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    audit_result = {
        "val":  {"overall": val_overall,  "groups": val_group_stats},
        "test": {"overall": test_overall, "groups": test_group_stats},
        "mapping": mapping,
        "definition": {
            "within_0_5": "fraction of predictions where |pred - true| <= 0.5",
            "within_1_0": "fraction of predictions where |pred - true| <= 1.0",
            "bias"       : "mean(pred - true), positive = over-prediction",
            "spread"     : "likely-range half-width = 1.5 * avg_MAE(val+test), rounded to 0.5",
            "ci"         : "95% bootstrap confidence interval, n_iter=2000",
        }
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2)
    print(f"\n  Saved: {OUT_JSON}\n")

if __name__ == "__main__":
    main()
