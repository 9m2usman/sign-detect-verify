"""Tests for FastAPI backend application routes and endpoints."""

from backend.app.main import app, health_check, root


def _resolve_coro(coro):
    """Resolve an un-awaited coroutine that returns synchronously."""
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value
    raise RuntimeError("Coroutine did not return immediately")


def test_app_metadata():
    """Verify application title, version, and routing structure."""
    assert app.title == "Signature Detection & Verification API"
    assert app.version == "1.0.0"
    route_paths = [route.path for route in app.routes]
    assert "/health" in route_paths
    assert "/" in route_paths


def test_health_check_endpoint():
    """Verify health check endpoint returns 200 OK and healthy status."""
    response = _resolve_coro(health_check())
    assert response["status"] == "healthy"
    assert response["service"] == "signature-backend"
    assert response["version"] == "1.0.0"


def test_root_endpoint():
    """Verify root endpoint returns welcome message and docs URL."""
    response = _resolve_coro(root())
    assert "message" in response
    assert response["docs_url"] == "/docs"
