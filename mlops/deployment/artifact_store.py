"""
Model Artifact Storage & Versioning Manager.
Handles model artifact packaging, SHA-256 checksum integrity verification,
metadata manifest generation, and versioned staging outside Git.
Zero Ultralytics, Zero AGPL.
"""

import os
import json
import hashlib
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Dict, Any, Optional


def compute_sha256(file_path: str) -> str:
    """Computes SHA-256 cryptographic hash of a model file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


class ModelArtifactStore:
    """
    Manages local and remote storage for versioned model artifacts outside Git.
    """
    def __init__(self, base_storage_dir: Optional[str] = None):
        if base_storage_dir:
            self.base_dir = base_storage_dir
        else:
            # Check if default storage path is writable, else fallback to tempdir
            target = os.path.abspath("storage/models")
            try:
                os.makedirs(target, exist_ok=True)
                test_file = os.path.join(target, f".test_{os.getpid()}")
                with open(test_file, "w") as f:
                    f.write("1")
                os.remove(test_file)
                self.base_dir = target
            except Exception:
                self.base_dir = os.path.join(tempfile.gettempdir(), "signature_model_artifacts")
                os.makedirs(self.base_dir, exist_ok=True)

    def publish_model(
        self,
        model_name: str,
        version_tag: str,
        weights_source_path: str,
        metrics: Dict[str, Any],
        framework: str = "PyTorch",
        license_name: str = "Apache-2.0 / BSD-3-Clause",
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Stores model weights, verifies integrity, and writes manifest.json.
        """
        if not os.path.exists(weights_source_path):
            raise FileNotFoundError(f"Source weights not found at: {weights_source_path}")

        sha256_hash = compute_sha256(weights_source_path)
        file_size_bytes = os.path.getsize(weights_source_path)

        version_dir = os.path.join(self.base_dir, model_name, version_tag)
        os.makedirs(version_dir, exist_ok=True)

        dest_weights_path = os.path.join(version_dir, os.path.basename(weights_source_path))
        shutil.copy2(weights_source_path, dest_weights_path)

        manifest = {
            "model_name": model_name,
            "version_tag": version_tag,
            "framework": framework,
            "license": license_name,
            "artifact_file": os.path.basename(weights_source_path),
            "sha256": sha256_hash,
            "size_bytes": file_size_bytes,
            "metrics": metrics,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "extra_metadata": extra_metadata or {}
        }

        manifest_path = os.path.join(version_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return {
            "status": "published",
            "model_name": model_name,
            "version_tag": version_tag,
            "weights_path": dest_weights_path,
            "manifest_path": manifest_path,
            "sha256": sha256_hash
        }

    def verify_artifact_integrity(self, model_name: str, version_tag: str) -> bool:
        """Validates that stored model file matches manifest SHA-256 hash."""
        version_dir = os.path.join(self.base_dir, model_name, version_tag)
        manifest_path = os.path.join(version_dir, "manifest.json")

        if not os.path.exists(manifest_path):
            return False

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        weights_path = os.path.join(version_dir, manifest["artifact_file"])
        if not os.path.exists(weights_path):
            return False

        current_hash = compute_sha256(weights_path)
        return current_hash == manifest["sha256"]
