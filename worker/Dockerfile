# Celery async worker Dockerfile
FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY worker /app/worker
COPY detection /app/detection
COPY verification /app/verification

ENV PYTHONPATH=/app

CMD ["celery", "-A", "worker.celery_app", "worker", "--loglevel=info", "--concurrency=2"]
