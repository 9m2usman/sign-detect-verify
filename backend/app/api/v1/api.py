"""
API Router Aggregator for Version 1.
"""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import signatures, jobs, models, webhooks

api_router = APIRouter()

api_router.include_router(signatures.router, prefix="/signatures", tags=["Signatures"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])
api_router.include_router(models.router, prefix="/models", tags=["Models"])
api_router.include_router(webhooks.router, prefix="/webhooks", tags=["Webhooks"])
