# CI/CD Pipeline Documentation

## 1. Pipeline Overview
Continuous Integration is executed on GitHub Actions (`.github/workflows/ci.yml`) on every push to `main` or pull request.

## 2. Pipeline Stages

1. **Environment Setup**:
   - Spawns PostgreSQL 16 Alpine and RabbitMQ 3.13 service containers with automated health checks.
   - Installs Python 3.11 with pip caching.
2. **Strict License & Copyleft Audit**:
   - Validates that no forbidden packages (such as `ultralytics`) or GPL-contaminated wheels are installed or imported.
3. **Automated Testing**:
   - Executes Unit tests: Model instantiation, preprocessing, loss calculation, schema validation.
   - Executes Integration tests: Database migrations, Celery task dispatch, API endpoints.
   - Asserts code coverage thresholds.
4. **Container Build & Security Scan**:
   - Builds backend and worker Docker images using multi-stage builds.
