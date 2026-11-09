"""
Quick-launch wrapper for Signature Detection & Verification API.
Ensures correct module paths regardless of which folder the terminal is in.
"""

import sys
import os

PROJECT_ROOT = "E:/sign_detect-verify"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)

import uvicorn

if __name__ == "__main__":
    print(f"🚀 Starting Uvicorn from project root: {PROJECT_ROOT}")
    uvicorn.run(
        "backend.app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )
