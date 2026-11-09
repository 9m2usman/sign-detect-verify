"""
Prometheus Observability & Application Metrics.
Tracks request rates, latency percentiles (p50, p95, p99), error rates,
signature detection counts, similarity score distributions, and genuine/forged decisions.
Provides graceful fallback when prometheus_client is not locally installed.
"""

import time
from typing import Callable, Dict, List, Any
from fastapi import Request, Response

try:
    from prometheus_client import (
        Counter as PromCounter,
        Histogram as PromHistogram,
        Gauge as PromGauge,
        generate_latest as prom_generate_latest,
        CONTENT_TYPE_LATEST
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class MockMetric:
        def __init__(self, name: str, description: str, labels: List[str] = None):
            self.name = name
            self.description = description
            self.label_names = labels or []
            self.values: Dict[str, float] = {}

        def labels(self, **kwargs):
            key = ",".join(f'{k}="{v}"' for k, v in sorted(kwargs.items()))
            class LabelProxy:
                def __init__(self, parent, k):
                    self.parent = parent
                    self.k = k
                def inc(self, amount: float = 1.0):
                    self.parent.values[self.k] = self.parent.values.get(self.k, 0.0) + amount
                def observe(self, value: float):
                    self.parent.values[self.k] = value
            return LabelProxy(self, key)

        def inc(self, amount: float = 1.0):
            self.values[""] = self.values.get("", 0.0) + amount

        def observe(self, value: float):
            self.values[f'val="{value}"'] = self.values.get(f'val="{value}"', 0.0) + 1.0

        def set(self, value: float):
            self.values[""] = value

    PromCounter = MockMetric
    PromHistogram = MockMetric
    PromGauge = MockMetric

    _registered_metrics = []

# 1. HTTP Ingestion Metrics
HTTP_REQUESTS_TOTAL = PromCounter(
    "http_requests_total",
    "Total count of HTTP requests handled by the API",
    ["method", "endpoint", "status_code"]
)

HTTP_REQUEST_DURATION_SECONDS = PromHistogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"]
)

# 2. Machine Learning Inference Metrics
SIGNATURE_DETECTIONS = PromHistogram(
    "signature_detections_per_document",
    "Distribution of detected signature bounding boxes per document"
)

SIGNATURE_SIMILARITY_SCORES = PromHistogram(
    "signature_similarity_score_distribution",
    "Distribution of biometric cosine similarity scores (0.0 to 1.0)"
)

SIGNATURE_DECISIONS_TOTAL = PromCounter(
    "signature_decisions_total",
    "Total signature authenticity decisions rendered",
    ["decision"]  # GENUINE, FORGED, REVIEW
)

# 3. Worker & Queue Metrics
WORKER_JOBS_PROCESSED_TOTAL = PromCounter(
    "worker_jobs_processed_total",
    "Total number of background verification jobs processed by Celery workers",
    ["status"]  # COMPLETED, FAILED
)

WORKER_JOB_DURATION_SECONDS = PromHistogram(
    "worker_job_duration_seconds",
    "End-to-end background job execution latency in seconds"
)

ACTIVE_WORKER_TASKS = PromGauge(
    "active_worker_tasks",
    "Current number of active tasks undergoing deep learning inference"
)

ALL_METRICS = [
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    SIGNATURE_DETECTIONS,
    SIGNATURE_SIMILARITY_SCORES,
    SIGNATURE_DECISIONS_TOTAL,
    WORKER_JOBS_PROCESSED_TOTAL,
    WORKER_JOB_DURATION_SECONDS,
    ACTIVE_WORKER_TASKS
]


async def prometheus_middleware(request: Request, call_next: Callable) -> Response:
    """Middleware recording HTTP request latency and throughput metrics."""
    if request.url.path == "/metrics":
        return await call_next(request)

    start_time = time.time()
    endpoint = request.url.path

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        status_code = 500
        raise
    finally:
        duration = time.time() - start_time
        try:
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=str(status_code)
            ).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)
        except Exception:
            pass

    return response


def metrics_endpoint_handler(request: Request = None) -> Response:
    """Exposes Prometheus text-formatted metrics."""
    if HAS_PROMETHEUS:
        data = prom_generate_latest()
    else:
        lines = []
        for m in ALL_METRICS:
            lines.append(f"# HELP {m.name} {m.description}")
            lines.append(f"# TYPE {m.name} untyped")
            if not m.values:
                lines.append(f"{m.name} 0.0")
            else:
                for k, v in m.values.items():
                    if k:
                        lines.append(f"{m.name}{{{k}}} {v}")
                    else:
                        lines.append(f"{m.name} {v}")
        data = ("\n".join(lines) + "\n").encode("utf-8")

    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
