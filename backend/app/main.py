"""
FastAPI Backend Application Entrypoint.
Initializes Database, CORS, Middleware, and API Routers.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

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

# Mount static asset directories if they exist
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

if os.path.exists("sample_test_run"):
    app.mount("/sample_test_run", StaticFiles(directory="sample_test_run"), name="sample_test_run")

if os.path.exists("architecture"):
    app.mount("/architecture", StaticFiles(directory="architecture"), name="architecture")

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


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


@app.get("/", tags=["UI"])
def root(request: Request = None):
    """Root endpoint: returns interactive HTML UI for browsers, or API dictionary."""
    if request is not None and "text/html" in request.headers.get("accept", ""):
        index_file = os.path.join(static_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs_url": "/docs",
        "ui_url": "/ui",
        "api_v1": settings.API_V1_STR,
        "license": "Apache-2.0 / BSD-3-Clause"
    }


@app.get("/ui", tags=["UI"])
def get_ui():
    """Explicit UI dashboard route serving interactive HTML5 frontend."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "UI not found"}

