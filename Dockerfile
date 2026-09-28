# ── RAG Chatbot (Groundwork) ────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first: this layer stays cached until requirements.txt changes,
# so code edits rebuild in seconds instead of minutes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code and the web UI
COPY app/ ./app/
COPY public/ ./public/

# Run as a non-root user; ChromaDB writes to /app/data (mount a volume there)
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

# Hosts like Render pass the port to listen on in $PORT; 8000 otherwise
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/api/health' % os.environ['PORT'], timeout=4)"

# 0.0.0.0 so the port is reachable from outside the container; no --reload in production.
# --proxy-headers: behind a host's load balancer, see the visitor's real IP and https.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --proxy-headers --forwarded-allow-ips "*"
