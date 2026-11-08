# Troubleshooting & Common Issues Guide

## 1. Broker Connection Errors

### Symptom: `celery.exceptions.OperationalError: [Errno 111] Connection refused`
- **Cause**: RabbitMQ is not running or broker credentials in `.env` are mismatched.
- **Remedy**:
  - Verify container status: `docker-compose ps rabbitmq`.
  - Check broker URL: default is `amqp://guest:guest@localhost:5672//`.
  - Inspect logs: `docker-compose logs rabbitmq`.

## 2. Database Connection Failures

### Symptom: `sqlalchemy.exc.OperationalError: could not connect to server`
- **Cause**: PostgreSQL service not ready or database does not exist.
- **Remedy**:
  - Verify PostgreSQL health: `docker-compose exec postgres pg_isready`.
  - Ensure `.env` parameters match `docker-compose.yml`.

## 3. CUDA Out of Memory (OOM)

### Symptom: `RuntimeError: CUDA out of memory`
- **Cause**: Batch size too large for GPU VRAM during detection inference.
- **Remedy**:
  - Reduce batch size in detection config (`detection/configs/faster_rcnn_config.yaml`).
  - Resize document images before inference to max dimension of 1024.
  - Run with CPU fallback by setting `DEVICE=cpu` in `.env`.
