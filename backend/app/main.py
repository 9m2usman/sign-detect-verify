"""
FastAPI Backend Application Entrypoint.
Initializes Database, CORS, Middleware, and API Routers.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.database.base import Base
from backend.app.database.session import engine
from backend.app.api.v1.api import api_router

# Create tables in development
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-grade REST microservice for signature object detection and metric verification.",
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1 routes
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint for Docker / ECS orchestration."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "active_detector": settings.ACTIVE_DETECTOR,
        "active_verifier": settings.ACTIVE_VERIFIER
    }


@app.get("/", tags=["Root"])
def root():
    """Root info endpoint with documentation links."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs_url": "/docs",
        "api_v1": settings.API_V1_STR,
        "license": "Apache-2.0 / BSD-3-Clause"
    }
