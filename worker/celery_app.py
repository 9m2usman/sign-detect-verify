"""
Celery Application Initialization with Graceful Local Fallback.
Configures RabbitMQ message broker, task queues, serialization, and retries.
When Celery is not installed or running locally, provides a transparent synchronous task fallback.
"""

import os
from backend.app.core.config import settings

try:
    from celery import Celery
    HAS_CELERY = True
except ImportError:
    HAS_CELERY = False
    
    class MockCeleryConf:
        def update(self, **kwargs):
            pass

    class MockCelery:
        def __init__(self, name: str, **kwargs):
            self.name = name
            self.conf = MockCeleryConf()

        def task(self, *args, **kwargs):
            def decorator(func):
                def delay(*a, **k):
                    return func(*a, **k)
                func.delay = delay
                return func
            return decorator

    Celery = MockCelery

celery_app = Celery(
    "signature_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["worker.tasks.inference_tasks"]
)

if HAS_CELERY:
    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_routes={
            "worker.tasks.inference_tasks.process_verification_job": {"queue": "inference_queue"},
            "worker.tasks.training_tasks.*": {"queue": "training_queue"},
        }
    )
