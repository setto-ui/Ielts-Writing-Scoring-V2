"""
app.py — Production REST API for IELTS Writing Scorer v2.2

Endpoints:
    POST /api/score  — Score an essay with full diagnostics & feedback
    POST /evaluate   — Backward-compatible endpoint
    GET  /health     — Health check & model metadata

Usage:
    python app.py
    gunicorn --bind 0.0.0.0:5000 --workers 2 app:app
"""

import re
import time
from pathlib import Path
from typing import Any
import numpy as np
from flask import Flask, request, jsonify

from evaluate import FEATURE_KEYS, extract_all_features, features_to_array
from scorer.normalizer import normalize_features
from scorer.ml_model import IELTSModel
from scorer.two_stage_model import TriSpecialistModel
from scorer.scorer import score_essay, round_to_ielts_band
from scorer.constraint import apply_constraints
from scorer.calibration import calibrate
from scorer.diagnostic_engine import analyze as analyze_diagnostic
from scorer.explainer import explain
from scorer.rewrite import rewrite_essay
from scorer.logger import build_pipeline_log
from scorer.tutor_engine import diagnose_errors, generate_band_8_rewrite

app = Flask(__name__)

# Model loading singletons
_model = None
_is_tri = False

MODEL_PATH_TRI = Path("data/model_tri_specialist.pkl")
MODEL_PATH_BASE = Path("data/model_exp_6JB.pkl")

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "i", "you", "it", "to", "of", "and",
    "in", "that", "this", "for", "on", "be", "with", "as", "have", "has", "had", "do",
    "does", "did", "not", "but", "or", "at", "by", "from", "they", "we", "he", "she",
    "some", "should", "what", "your", "extent", "agree", "disagree", "give", "write",
    "least", "words", "discuss", "both", "views", "own", "opinion", "why", "how",
    "many", "people", "think", "believe", "others", "reasons", "answer", "include",
    "any", "relevant", "examples", "knowledge", "experience"
}


def get_model():
    global _model, _is_tri
    if _model is None:
        if MODEL_PATH_TRI.exists():
            try:
                _model = TriSpecialistModel.load(MODEL_PATH_TRI)
                _is_tri = True
                print(f"[OK] Tri-Specialist Two-Stage Model loaded from {MODEL_PATH_TRI}")
            except Exception as e:
                print(f"[WARN] Failed loading Tri-Specialist ({e}), falling back to baseline.")
        
        if _model is None and MODEL_PATH_BASE.exists():
            try:
                _model = IELTSModel.load(MODEL_PATH_BASE)
                _is_tri = False
                print(f"[OK] Baseline IELTSModel loaded from {MODEL_PATH_BASE}")
            except Exception as e:
                print(f"[ERROR] Failed loading baseline model: {e}")

    return _model, _is_tri


def compute_keyword_analysis(prompt: str, essay: str) -> dict[str, Any]:
    prompt_words = re.findall(r"\b[a-z']+\b", prompt.lower())
    prompt_kws = sorted({w for w in prompt_words if w not in _STOPWORDS and len(w) > 3})
    essay_words = set(re.findall(r"\b[a-z']+\b", essay.lower()))
    matched = sorted(set(prompt_kws) & essay_words)
    missing = sorted(set(prompt_kws) - essay_words)
    coverage = len(matched) / len(prompt_kws) * 100 if prompt_kws else 100.0
    return {
        "topic_coverage_pct": round(coverage, 1),
        "total_prompt_keywords": len(prompt_kws),
        "used_keywords": matched,
        "missing_keywords": missing,
        "is_off_topic_warning": coverage < 50.0
    }


def compute_confidence_stats(overall: float) -> dict[str, Any]:
    if overall <= 5.5:
        return {
            "level": "LOW",
            "accuracy_pct": 30,
            "likely_range": [max(4.0, overall - 1.0), min(9.0, overall + 1.0)],
            "note": "Low band range. Actual band may vary by 0.5-1.0."
        }
    elif overall <= 7.0:
        return {
            "level": "HIGH",
            "accuracy_pct": 86,
            "likely_range": [max(4.0, overall - 0.5), min(9.0, overall + 0.5)],
            "note": "Most reliable range. Locked test accuracy 86.3% within +-0.5 band."
        }
    else:
        return {
            "level": "MEDIUM",
            "accuracy_pct": 40,
            "likely_range": [max(4.0, overall - 1.0), min(9.0, overall + 1.0)],
            "note": "High band range. Advanced vocabulary & structure detected; actual may vary +-1.0."
        }


# Warm up model on startup
get_model()


@app.route("/health", methods=["GET"])
def health():
    model, is_tri = get_model()
    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "model_type": "Tri-Specialist Two-Stage v2.2" if is_tri else "Baseline LightGBM",
        "features_count": len(FEATURE_KEYS),
        "tri_model_loaded": is_tri
    }), 200


@app.route("/api/score", methods=["POST"])
@app.route("/evaluate", methods=["POST"])
def evaluate():
    model, is_tri = get_model()
    if model is None:
        return jsonify({"error": "No trained model found. Check data/ directory."}), 503

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    essay = payload.get("essay", "").strip()
    prompt = payload.get("prompt", "").strip()

    words = essay.split()
    if len(words) < 20:
        return jsonify({"error": "Essay too short (minimum 20 words required for assessment)"}), 400
    if not prompt:
        return jsonify({"error": "Exam prompt is required for scoring"}), 400

    t0 = time.time()

    # 1. Feature Extraction (31 features)
    raw_features = extract_all_features(essay, prompt)
    norm_features = normalize_features(raw_features)

    # 2. ML Prediction
    X = features_to_array(norm_features, FEATURE_KEYS).reshape(1, -1)
    ml_raw = model.predict(X)
    ml_preds = {k: float(np.squeeze(v)) for k, v in ml_raw.items()}

    # 3. Hard Gates & Criteria
    score_result = score_essay(
        normalized_features=norm_features,
        raw_features=raw_features,
        ml_predictions=ml_preds,
    )
    gated_scores = score_result["scores"]
    gate_names = score_result.get("gates", [])

    # 4. Constraints Layer
    constrained_scores, applied_rules = apply_constraints(gated_scores, raw_features)

    # 5. Overall Band Calculation
    if is_tri:
        overall_band = round_to_ielts_band(float(constrained_scores.get("overall", ml_preds["overall"])))
    else:
        raw_mean = float(np.mean(list(constrained_scores.values())))
        overall_band = calibrate(raw_mean)

    rounded_subscores = {k: round_to_ielts_band(v) for k, v in constrained_scores.items()}

    # 6. Diagnostics & Feedback
    diagnostic_result = analyze_diagnostic(
        normalized_features=norm_features,
        raw_features=raw_features,
        overall_pred=overall_band,
        is_estimated=score_result.get("is_estimated", False),
        sub_scores=rounded_subscores,
    )

    feedback = explain(
        diagnostic_analysis=diagnostic_result,
        features=raw_features,
        applied_constraints=gate_names + applied_rules,
        gaming_flags=raw_features.get("_gaming_flags", {}),
    )

    rewrite_result = rewrite_essay(essay, diagnostic_result)
    tutor_diagnosis = diagnose_errors(essay, prompt)
    model_rewrite = generate_band_8_rewrite(essay, prompt, tutor_diagnosis)
    keyword_analysis = compute_keyword_analysis(prompt, essay)
    confidence = compute_confidence_stats(overall_band)

    elapsed_ms = round((time.time() - t0) * 1000, 1)

    response = {
        "status": "success",
        "overall_band": overall_band,
        "sub_scores": {
            "task_response": rounded_subscores.get("task_response"),
            "coherence_cohesion": rounded_subscores.get("coherence"),
            "lexical_resource": rounded_subscores.get("lexical"),
            "grammatical_range_accuracy": rounded_subscores.get("grammar"),
        },
        "word_count": int(raw_features.get("word_count", len(words))),
        "task_type": raw_features.get("task_type", "unknown"),
        "keyword_analysis": keyword_analysis,
        "confidence": confidence,
        "constraints_applied": applied_rules,
        "feedback": {
            "strengths": feedback.get("strengths", []),
            "weaknesses": feedback.get("weaknesses", []),
            "actionable_tips": feedback.get("actionable_tips", []),
            "detailed_critique": feedback.get("summary", ""),
        },
        "tutor_analysis": {
            "error_diagnosis": tutor_diagnosis,
            "model_rewrite_band_8": model_rewrite["model_essay"],
            "key_upgrades": model_rewrite["key_upgrades"]
        },
        "rewrite_suggestions": rewrite_result,
        "meta": {
            "model_version": "v2.2-tri-specialist" if is_tri else "v2.0-baseline",
            "latency_ms": elapsed_ms,
        }
    }

    return jsonify(response), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
