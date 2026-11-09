# Multi-stage Dockerfile for Automated Remote Retraining Job
# Zero AGPL, Non-root User, Minimal Attack Surface

FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

COPY training /app/training
COPY detection /app/detection
COPY verification /app/verification
COPY mlops /app/mlops
COPY backend /app/backend

RUN mkdir -p /app/storage /app/checkpoints && \
    chown -R appuser:appgroup /app

USER appuser

ENTRYPOINT ["python", "-m", "training.train_both_models_local"]
