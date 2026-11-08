"""Unit tests verifying project scaffolding, licensing compliance, and configuration integrity."""

import os
from pathlib import Path


def test_no_ultralytics_in_requirements():
    """Ensure ultralytics is strictly absent from requirements.txt."""
    req_file = Path("requirements.txt")
    assert req_file.exists(), "requirements.txt must exist"
    content = req_file.read_text(encoding="utf-8").lower()
    assert "ultralytics" not in content, "ultralytics must NOT be present in requirements.txt"


def test_required_dependencies_in_requirements():
    """Ensure all required permissive dependencies are declared."""
    req_file = Path("requirements.txt")
    content = req_file.read_text(encoding="utf-8").lower()
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "sqlalchemy",
        "celery",
        "redis",
        "mlflow",
        "torch",
        "torchvision",
        "opencv-python-headless",
        "pillow",
        "numpy",
        "pytest",
        "requests",
    ]
    for pkg in required_packages:
        assert pkg in content, f"Package {pkg} must be in requirements.txt"


def test_gitignore_preserves_datasets():
    """Ensure .gitignore excludes large datasets and model weights."""
    gitignore = Path(".gitignore")
    assert gitignore.exists(), ".gitignore must exist"
    content = gitignore.read_text(encoding="utf-8")
    assert "merged_signature_dataset.zip" in content
    assert "merged_signature_dataset/" in content
    assert "data/raw/" in content
    assert "data/processed/" in content
    assert "*.pt" in content
    assert "*.pth" in content
    assert "*.onnx" in content
    assert ".env" in content
    assert "__pycache__/" in content
    assert ".venv/" in content
    assert "mlruns/" in content


def test_docker_compose_services():
    """Ensure docker-compose.yml configures required services and ports."""
    dc_file = Path("docker-compose.yml")
    assert dc_file.exists(), "docker-compose.yml must exist"
    content = dc_file.read_text(encoding="utf-8")
    assert "postgres:" in content
    assert "5432:5432" in content
    assert "rabbitmq:" in content
    assert "5672:5672" in content
    assert "15672:15672" in content
    assert "mlflow:" in content
    assert "5000:5000" in content
    assert "backend:" in content
    assert "8000:8000" in content
    assert "worker:" in content


def test_env_example_template():
    """Ensure .env.example contains required template environment variables."""
    env_file = Path(".env.example")
    assert env_file.exists(), ".env.example must exist"
    content = env_file.read_text(encoding="utf-8")
    assert "POSTGRES_DB=" in content
    assert "RABBITMQ_HOST=" in content
    assert "CELERY_BROKER_URL=" in content
    assert "MLFLOW_TRACKING_URI=" in content


def test_dataset_master_preservation():
    """Ensure master dataset archive and directory are preserved intact."""
    zip_path = Path("merged_signature_dataset.zip")
    dir_path = Path("merged_signature_dataset")
    assert zip_path.exists(), "merged_signature_dataset.zip must exist"
    assert zip_path.stat().st_size > 3_000_000_000, "Archive size should be ~3.26 GB"
    assert dir_path.exists() and dir_path.is_dir(), "merged_signature_dataset directory must exist"


def test_part17_directory_scaffolding():
    """Ensure all required directories from Part 17 are created."""
    expected_dirs = [
        "backend/app/api",
        "backend/app/core",
        "backend/app/database",
        "backend/app/schemas",
        "backend/app/services",
        "backend/app/models",
        "backend/tests",
        "worker/tasks",
        "worker/services",
        "detection/configs",
        "detection/preprocessing",
        "detection/training",
        "detection/evaluation",
        "detection/experiments",
        "detection/selected_model",
        "verification/preprocessing",
        "verification/models",
        "verification/training",
        "verification/evaluation",
        "verification/experiments",
        "verification/selected_model",
        "notebooks/detection",
        "notebooks/verification",
        "notebooks/evaluation",
        "mlops/tracking",
        "mlops/registry",
        "mlops/webhooks",
        "mlops/training_pipeline",
        "mlops/deployment",
        "infrastructure/docker",
        "infrastructure/aws",
        "infrastructure/github-actions",
        "tests/unit",
        "tests/integration",
        "tests/end_to_end",
        "data/raw",
        "data/processed",
        "architecture",
        ".github/workflows",
    ]
    for d in expected_dirs:
        p = Path(d)
        assert p.exists() and p.is_dir(), f"Required directory {d} does not exist"


def test_documentation_files_exist():
    """Ensure all required documentation files exist and are not empty."""
    doc_files = [
        "requirements.md",
        "research_references.md",
        "dataset_documentation.md",
        "model_licensing.md",
        "detection_evaluation.md",
        "detection_model_selection.md",
        "verification_evaluation.md",
        "verification_model_selection.md",
        "architecture.md",
        "api_documentation.md",
        "mlops_documentation.md",
        "ci_cd_documentation.md",
        "aws_deployment.md",
        "manual_testing_guide.md",
        "troubleshooting.md",
        "user_guide.md",
    ]
    for doc in doc_files:
        p = Path("docs") / doc
        assert p.exists(), f"docs/{doc} must exist"
        assert p.stat().st_size > 100, f"docs/{doc} must not be empty"


def test_fastapi_app_structure():
    """Ensure FastAPI backend main application can be imported and configured."""
    from backend.app.main import app

    assert app.title == "Signature Detection & Verification API"
    routes = [route.path for route in app.routes]
    assert "/health" in routes
    assert "/" in routes


def test_pytest_configuration_avoids_dataset_crawl():
    """Ensure pytest.ini prevents test collection from crawling the 57k file dataset."""
    pytest_ini = Path("pytest.ini")
    assert pytest_ini.exists(), "pytest.ini must exist"
    content = pytest_ini.read_text(encoding="utf-8")
    assert "norecursedirs" in content
    assert "merged_signature_dataset" in content
