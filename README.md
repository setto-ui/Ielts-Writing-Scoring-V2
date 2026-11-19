# IELTS Writing Scorer v2.2 (Tri-Specialist Two-Stage)

An automated IELTS Writing Task 2 scoring, diagnostic, and tutoring engine. Built with **LightGBM**, **spaCy**, **SentenceTransformers**, and a **Tri-Specialist Soft-Blending Architecture** designed to eliminate extreme-band regression-to-the-mean.

Runs **100% offline and local** with no paid API keys required.

---

## 🌟 Key Highlights & Benchmark Results

### Locked Test Set Performance ($N=191$, `data/test.csv`)

| Metric | Baseline Single LightGBM | **Tri-Specialist v2.2** | Improvement |
|---|---|---|---|
| **Overall MAE** | 0.775 | **0.686** | **-0.089 band error reduction** |
| **Overall RMSE** | 1.021 | **0.904** | **-0.117** |
| **$\pm 0.5$ Band Accuracy** | 55.0% | **61.8%** | **+6.8% accuracy increase** |
| **$\pm 1.0$ Band Accuracy** | 77.5% | **83.8%** | **+6.3% accuracy increase** |
| **Overall Bias** | $+0.162$ | **$-0.037$** | **Near-zero bias** |
| **Core Mid Range ($6.0 - 7.0$) Acc** | 71.6% | **86.3%** | **Record 86.3% accuracy (MAE 0.363)** |
| **Low Band ($4.0 - 5.5$) MAE** | 1.226 | **1.048** | **Significant regression-to-mean fix** |

---

## 🏗️ System Architecture

```text
               User Essay + Exam Prompt
                         │
                         ▼
             31 Linguistic Features
        (Lexical, Syntactic, Structural, Cohesion)
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
[Generalist Base]  [Low Detector]   [High Detector]
 LightGBM Regressor  P(Band <= 5.5)   P(Band >= 7.0)
        │                │                │
        │                ▼                ▼
        │          [Low Regressor]  [High Regressor]
        │                │                │
        └────────────────┼────────────────┘
                         ▼
        Expected Value Soft Blending Engine
                         │
                         ▼
             Hard Gates & Constraint Layer
                         │
                         ▼
             Diagnostic & Feedback Engine
        (Sub-scores, Strengths, Actionable Tips)
```

### Why Tri-Specialist Soft Blending?
In standard single-regressor models trained on imbalanced essay corpora (where Band 6.0–7.0 represents ~73% of data), extreme essays are forcefully pulled toward the mean (Band 8 scores become 7.0, Band 4 scores become 5.5).

The **Tri-Specialist v2.2** system solves this mathematically:
1. **Generalist Model**: Captures standard essay distributions with high stability.
2. **Balanced Binary Classifiers**: Inverse-frequency weighted detectors compute honest tail probabilities $P(\text{Low})$ and $P(\text{High})$.
3. **Specialist Regressors**: Extrapolate vocabulary sophistication and syntactic complexity for extreme bands.
4. **Continuous Soft Formulation**: Avoids discrete cliff-edge classification errors.

---

## 🚀 Quick Start (Local Setup)

### 1. Prerequisites
- Python 3.10 or 3.11
- Java 17+ (Optional, recommended for LanguageTool grammar engine)

### 2. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd "Ielts writing scoring v2"

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
```

### 3. Interactive CLI Scorer
Test any prompt and essay directly in your terminal:
```bash
python test_scorer.py
```
*Type `end` then Enter to submit prompt and essay.*

### 4. Run the REST API Server
```bash
python app.py
```
*API server will listen on `http://localhost:5000`.*

---

## 🐳 Docker Deployment

The provided Docker container includes OpenJDK 17 and pre-caches the embedding model during build time so it can run completely offline.

### Using Docker Compose (Recommended)

```bash
# Build and run the REST API container in background
docker compose up -d --build

# View logs
docker compose logs -f

# Check health
curl http://localhost:5000/health
```

### Interactive CLI inside Docker
```bash
docker compose run --rm ielts-scorer-cli
```

### Manual Docker Build & Run
```bash
docker build -t ielts-writing-scorer:v2.2 .
docker run -d -p 5000:5000 --name ielts_scorer ielts-writing-scorer:v2.2
```

---

## 📡 REST API Reference

### 1. Score Essay (`POST /api/score` or `POST /evaluate`)

**Request:**
```bash
curl -X POST http://localhost:5000/api/score \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Some people think that children should start learning a foreign language at primary school rather than secondary school. Do the advantages outweigh the disadvantages?",
    "essay": "In today'\''s globalized world, the question of when children should begin learning a foreign language has become a topic of significant debate. While some argue that early exposure at primary school is beneficial, others believe that secondary school is the more appropriate time to start. In my opinion, introducing foreign languages at primary school offers greater long-term advantages..."
  }'
```

**Response:**
```json
{
  "status": "success",
  "overall_band": 7.5,
  "sub_scores": {
    "task_response": 8.0,
    "coherence_cohesion": 7.5,
    "lexical_resource": 7.5,
    "grammatical_range_accuracy": 6.5
  },
  "word_count": 296,
  "task_type": "advantages_outweigh",
  "keyword_analysis": {
    "topic_coverage_pct": 100.0,
    "total_prompt_keywords": 11,
    "used_keywords": ["children", "foreign", "language", "learning", "primary", "school"],
    "missing_keywords": [],
    "is_off_topic_warning": false
  },
  "confidence": {
    "level": "MEDIUM",
    "accuracy_pct": 40,
    "likely_range": [6.5, 8.5],
    "note": "High band range. Advanced vocabulary & structure detected; actual may vary +-1.0."
  },
  "feedback": {
    "strengths": ["Well-developed paragraphs with clear topic sentences"],
    "weaknesses": ["Minor tense inconsistency in paragraph 2"],
    "actionable_tips": ["Expand complex clause variety in supporting arguments"]
  },
  "meta": {
    "model_version": "v2.2-tri-specialist",
    "latency_ms": 320.5
  }
}
```

### 2. Healthcheck (`GET /health`)
```json
{
  "status": "healthy",
  "model_type": "Tri-Specialist Two-Stage v2.2",
  "features_count": 31,
  "tri_model_loaded": true
}
```

---

## 📁 Repository Structure

```
├── scorer/                     # Core evaluation & NLP engine
│   ├── two_stage_model.py      # Tri-Specialist architecture (v2.2 champion)
│   ├── ml_model.py             # Single LightGBM model wrapper
│   ├── features_text.py        # Syntactic & lexical feature extraction
│   ├── features_essay.py       # Structural & coherence feature extraction
│   ├── features_task.py        # Task coverage & prompt overlap
│   ├── features_lexical.py     # Academic word ratio & lexical diversity
│   ├── features_antigaming.py  # Gaming penalty & connector spam detection
│   ├── scorer.py               # Criterion hard-gating & aggregation
│   ├── constraint.py           # Safety constraints (word count penalty, etc.)
│   └── calibration.py          # Band score calibration
├── data/                       # Datasets & trained model binaries
│   ├── model_tri_specialist.pkl # Champion Two-Stage production model
│   ├── model_exp_6JB.pkl       # Baseline LightGBM model
│   ├── feature_stats.json      # Z-score normalization statistics
│   ├── train.csv               # Training dataset (N=886)
│   ├── val.csv                 # Validation dataset (N=190)
│   └── test.csv                # Locked test dataset (N=191)
├── app.py                      # Production Flask REST API
├── test_scorer.py              # Interactive CLI scoring interface
├── evaluate.py                 # Ground-truth evaluation pipeline
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Container orchestration
└── README.md
```

---

## 📜 License
Educational and research use only.
