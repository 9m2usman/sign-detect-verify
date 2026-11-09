# calary working
(.venv) PS E:\sign_detect-verify> cd E:\sign_detect-verify
>> .\.venv\Scripts\Activate.ps1
>> celery -A worker.celery_app:celery_app worker --loglevel=INFO --pool=solo

# mlflow running
(.venv) PS E:\sign_detect-verify> cd E:\sign_detect-verify
>> .\.venv\Scripts\Activate.ps1
>> 
>> mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///E:/sign_detect-verify/mlflow.db --default-artifact-root E:/sign_detect-verify/mlartifacts

# backend running
(.venv) PS E:\sign_detect-verify> cd E:\sign_detect-verify
 .\.venv\Scripts\Activate.ps1
 uvicorn backend.app.main:app --host 127.0.0.1 --port 8000