"""
Tests for FastAPI backend application routes and endpoints.
"""

from backend.app.main import app, health_check, root
from backend.app.core.config import settings


def test_app_metadata():
    """Verify application title, version, and routing structure."""
    assert app.title == settings.PROJECT_NAME
    assert app.version == settings.VERSION
    route_paths = [route.path for route in app.routes if hasattr(route, "path")]
    assert "/health" in route_paths
    assert "/" in route_paths


def test_health_check_endpoint():
    """Verify health check endpoint returns healthy status."""
    response = health_check()
    assert response["status"] == "healthy"
    assert response["service"] == settings.PROJECT_NAME
    assert response["version"] == settings.VERSION


def test_root_endpoint():
    """Verify root endpoint returns welcome message and docs URL."""
    response = root()
    assert "message" in response
    assert response["docs_url"] == "/docs"
