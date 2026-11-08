"""FastAPI backend application entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Signature Detection & Verification API",
    description="Permissive, production-grade microservice for document signature detection and Siamese verification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for Docker / orchestration."""
    return {"status": "healthy", "service": "signature-backend", "version": "1.0.0"}


@app.get("/", tags=["Root"])
async def root():
    """Root info endpoint."""
    return {
        "message": "Welcome to Signature Detection & Verification API",
        "docs_url": "/docs",
    }
