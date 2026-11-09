# Multi-stage Hardened Dockerfile for Celery Inference Worker
# Zero AGPL, Non-root User, Minimal Attack Surface

# ==========================================
# Stage 1: Build Dependencies
# ==========================================
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ==========================================
# Stage 2: Runtime Image
# ==========================================
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create unprivileged application user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy application source files & model modules
COPY backend /app/backend
COPY worker /app/worker
COPY detection /app/detection
COPY verification /app/verification
COPY training /app/training
COPY mlops /app/mlops
COPY checkpoints /app/checkpoints

# Setup writable directories with correct permissions for non-root user
RUN mkdir -p /app/uploads /app/storage /app/data /app/checkpoints && \
    chown -R appuser:appgroup /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "from worker.celery_app import celery_app; res = celery_app.control.ping(timeout=2.0); exit(0 if res else 1)" || exit 1

CMD ["celery", "-A", "worker.celery_app", "worker", "--loglevel=info", "--concurrency=2", "-Q", "inference_queue,celery"]
