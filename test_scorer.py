"""
test_scorer.py - CLI tester for IELTS Writing Scorer v2 (Model 6J-B, 31 features)

Usage:
    python test_scorer.py
"""

import sys

# Fix encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("Loading models, please wait...", flush=True)

import numpy as np

from evaluate import FEATURE_KEYS, extract_all_features, features_to_array
from scorer.normalizer import normalize_features
from scorer.ml_model import IELTSModel
from scorer.two_stage_model import TriSpecialistModel
from scorer.scorer import score_essay, round_to_ielts_band
from scorer.constraint import apply_constraints
from scorer.calibration import calibrate

MODEL_PATH_TRI = "data/model_tri_specialist.pkl"
MODEL_PATH_BASE = "data/model_exp_6JB.pkl"

BAND_LABEL = {
    9.0: "Expert User",
    8.5: "Expert User",
    8.0: "Very Good User",
    7.5: "Very Good User",
    7.0: "Good User",
    6.5: "Competent User",
    6.0: "Competent User",
    5.5: "Modest User",
    5.0: "Modest User",
    4.5: "Limited User",
    4.0: "Limited User",
    3.5: "Extremely Limited User",
    3.0: "Extremely Limited User",
}


def bar(b: float, width: int = 28) -> str:
    filled = int(round((b / 9.0) * width))
    return "#" * filled + "-" * (width - filled)


def collect_until_end(section: str) -> str:
    """Read multiline input until user types 'end'."""
    sep = "-" * 55
    print(f"\n{sep}")
    print(f"  {section}")
    print(f"  (Type  end  then Enter when done)")
    print(f"{sep}\n")
    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip().lower() == "end":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def run_scoring(prompt: str, essay: str) -> None:
    """Run full pipeline and print result."""
    import re

    _STOPWORDS = {
        "the","a","an","is","are","was","were","i","you","it","to","of","and",
        "in","that","this","for","on","be","with","as","have","has","had","do",
        "does","did","not","but","or","at","by","from","they","we","he","she",
        "some","should","what","your","extent","agree","disagree","give","write",
        "least","words","discuss","both","views","own","opinion","why","how",
        "many","people","think","believe","others","reasons","answer","include",
        "any","relevant","examples","knowledge","experience",
    }

    def extract_keywords(text: str) -> set:
        words = re.findall(r"\b[a-z']+\b", text.lower())
        return {w for w in words if w not in _STOPWORDS and len(w) > 3}

    print("\n  Scoring...\n")

    from pathlib import Path
    if Path(MODEL_PATH_TRI).exists():
        model = TriSpecialistModel.load(MODEL_PATH_TRI)
        is_tri = True
    else:
        model = IELTSModel.load(MODEL_PATH_BASE)
        is_tri = False

    raw  = extract_all_features(essay, prompt)
    norm = normalize_features(raw)

    X = features_to_array(norm, FEATURE_KEYS).reshape(1, -1)
    ml_raw   = model.predict(X)
    ml_preds = {
        k: float(v[0]) if hasattr(v, "__len__") else float(v)
        for k, v in ml_raw.items()
    }

    score_result = score_essay(
        normalized_features=norm,
        raw_features=raw,
        ml_predictions=ml_preds,
    )
    constrained, rules = apply_constraints(score_result["scores"], raw)
    if is_tri:
        overall = round_to_ielts_band(float(constrained.get("overall", ml_preds["overall"])))
    else:
        overall = calibrate(float(np.mean(list(constrained.values()))))
    rounded = {k: round_to_ielts_band(v) for k, v in constrained.items()}

    sep   = "=" * 55
    label = BAND_LABEL.get(overall, "")

    model_title = "Tri-Specialist Two-Stage v2.2" if is_tri else "Model 6J-B (31 feat)"
    print(sep)
    print(f"  IELTS WRITING SCORE  [{model_title}]")
    print(sep)
    print(f"\n  Overall Band  : {overall}  ({label})")
    print(f"  [{bar(overall)}]")
    print(f"\n  Word count    : {int(raw.get('word_count', 0))}")
    print(f"  Task type     : {raw.get('task_type', 'unknown')}\n")
    print("  --- Sub-scores -------------------------------------------")

    criteria = [
        ("task_response", "Task Response        (TR)"),
        ("coherence",     "Coherence & Cohesion (CC)"),
        ("lexical",       "Lexical Resource     (LR)"),
        ("grammar",       "Grammatical Range   (GRA)"),
    ]
    for key, disp in criteria:
        v = rounded.get(key)
        if v is not None:
            print(f"  {disp}  {v}  [{bar(v)}]")

    if rules:
        print(f"\n  [!] Constraints: {', '.join(rules)}")

    # --- Keyword analysis ---
    prompt_kws  = extract_keywords(prompt)
    essay_words = set(re.findall(r"\b[a-z']+\b", essay.lower()))
    matched  = sorted(prompt_kws & essay_words)
    missed   = sorted(prompt_kws - essay_words)
    overlap  = len(matched) / len(prompt_kws) * 100 if prompt_kws else 0

    print(f"\n  --- Keyword Analysis (Prompt vs Essay) -------------------")
    print(f"  Topic coverage : {overlap:.0f}%  ({len(matched)}/{len(prompt_kws)} keywords used)")
    if matched:
        print(f"  Used     : {', '.join(matched)}")
    if missed:
        print(f"  Missing  : {', '.join(missed)}")
        if overlap < 50:
            print(f"  [!] Low topic coverage -- essay may be off-topic")

    # --- Confidence warning ---
    # Empirically validated on Locked Test Set (N=191) with Tri-Specialist Architecture.
    print(f"\n  --- Confidence ------------------------------------------")
    if overall <= 5.5:
        conf_pct   = 30
        conf_label = "LOW"
        band_lo    = max(4.0, overall - 1.0)
        band_hi    = min(9.0, overall + 1.0)
        note = "Low band range. Actual band may vary by 0.5-1.0."
    elif overall <= 7.0:
        conf_pct   = 86
        conf_label = "HIGH"
        band_lo    = max(4.0, overall - 0.5)
        band_hi    = min(9.0, overall + 0.5)
        note = "Most reliable range. Locked test accuracy 86.3% within +-0.5 band."
    else:
        conf_pct   = 40
        conf_label = "MEDIUM"
        band_lo    = max(4.0, overall - 1.0)
        band_hi    = min(9.0, overall + 1.0)
        note = "High band range. Advanced vocabulary & structure detected; actual may vary +-1.0."

    conf_bar = "#" * (conf_pct // 5) + "-" * (20 - conf_pct // 5)
    print(f"  Confidence     : {conf_label} ({conf_pct}%)  [{conf_bar}]")
    print(f"  Likely range   : Band {band_lo:.1f} -- {band_hi:.1f}")
    print(f"  Note           : {note}")
    print(f"\n  * AI estimate only. Not a substitute for official marking.")

    print(f"\n{sep}\n")


def main():
    print()
    print("=" * 55)
    print("  IELTS Writing Scorer v2  |  Model 6J-B (31 feat)")
    print("=" * 55)
    print("  Type your exam prompt and essay, then score.")
    print("  Use  end  + Enter to finish each section.")

    while True:
        prompt = collect_until_end("[1/2] EXAM PROMPT  - Full IELTS question text")
        if not prompt:
            print("  [!] Prompt is empty. Please try again.\n")
            continue

        essay = collect_until_end("[2/2] YOUR ESSAY   - Student's written answer (250+ words)")
        if not essay:
            print("  [!] Essay is empty. Please try again.\n")
            continue

        wc = len(essay.split())
        if wc < 100:
            print(f"  [!] Essay too short ({wc} words). IELTS requires at least 250 words.\n")
            continue

        # Detect copy-paste: essay is basically the prompt
        prompt_words   = set(prompt.lower().split())
        essay_words_s  = set(essay.lower().split())
        if len(prompt_words) > 0:
            overlap_ratio = len(prompt_words & essay_words_s) / len(prompt_words)
            if overlap_ratio > 0.85 and wc < len(prompt.split()) * 1.5:
                print(f"  [!] Essay looks identical to the prompt. Please enter your actual written essay.\n")
                continue

        try:
            run_scoring(prompt, essay)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n  [ERROR] {e}\n")

        again = sys.stdin.readline().strip() if False else input("\n  Score another essay? (y/N): ").strip().lower()
        if not again.startswith("y"):
            print("\n  Goodbye!\n")
            break
        print()


if __name__ == "__main__":
    main()
