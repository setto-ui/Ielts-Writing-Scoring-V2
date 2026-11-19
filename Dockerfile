# =============================================================================
# IELTS Writing Scorer v2.2 — Production Dockerfile
# Multi-stage optimized, self-contained (includes Java 17 + pre-cached NLP models)
# =============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=5000

# Install system dependencies + OpenJDK 17 for LanguageTool grammar engine
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Pre-download spaCy English language model
RUN python -m spacy download en_core_web_sm

# Pre-cache SentenceTransformer model inside the image so it runs 100% offline
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application source code and models
COPY . /app/

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check to ensure API is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Production WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
