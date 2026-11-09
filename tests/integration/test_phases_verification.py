"""
Comprehensive Integration Tests for Master Prompt Phases 2-7.
Verifies:
- Accurate signature localization (tight bounding boxes, no full-page box).
- Calibrated Siamese verification (authentic > 0.85, impostor < 0.55).
- Multi-reference handling with individual and aggregated scores.
- Crop URLs returning web-accessible paths (/uploads/...).
- Database persistence and job status retrieval.
"""

import os
from pathlib import Path
import pytest
from PIL import Image
from fastapi import UploadFile

from backend.app.api.v1.endpoints.signatures import verify_signature_endpoint
from backend.app.api.v1.endpoints.jobs import get_job_status, to_web_url
from backend.app.core.config import settings
from backend.app.database.session import SessionLocal
from detection.selected_model.detector import ProductionSignatureDetector
from verification.selected_model.verifier import ProductionSignatureVerifier

BASE_DIR = Path("E:/sign_detect-verify")


def test_phase3_detector_tight_bounding_box():
    """Verify detector extracts real signature on sample_document.jpg without full-page hallucination."""
    doc_path = str(BASE_DIR / "sample_test_run/sample_document.jpg")
    assert os.path.exists(doc_path), f"Sample document not found: {doc_path}"

    img = Image.open(doc_path)
    w, h = img.size

    detector = ProductionSignatureDetector()
    detections = detector.detect(doc_path)

    assert len(detections) >= 1, "Expected at least 1 signature detection"
    primary_bbox = detections[0]["bbox"]
    x1, y1, x2, y3 = primary_bbox
    bw = x2 - x1
    bh = y3 - y1
    box_area = bw * bh
    total_area = w * h

    # Bounding box must be local to the signature, NOT the whole document
    assert box_area < 0.35 * total_area, f"Box area too large ({box_area}/{total_area}), possible poisoned box"
    # Signature is in lower half of sample_document.jpg
    assert y1 > 0.40 * h, f"Signature top ({y1}) should be in lower half of document ({h})"
    assert detections[0]["confidence"] >= 0.80


def test_phase4_siamese_verifier_accuracy_and_separation():
    """Verify authentic pair scores > 0.85 (GENUINE) and impostor pair scores < 0.55 (FORGED)."""
    verifier = ProductionSignatureVerifier()
    assert verifier.model is not None, "SiameseSignatureNet model failed to load weights"

    auth_path = str(BASE_DIR / "sample_test_run/reference_authentic.jpg")
    imp_path = str(BASE_DIR / "sample_test_run/reference_impostor.jpg")
    doc_path = str(BASE_DIR / "sample_test_run/sample_document.jpg")

    assert os.path.exists(auth_path)
    assert os.path.exists(imp_path)
    assert os.path.exists(doc_path)

    doc_img = Image.open(doc_path)
    crop = doc_img.crop((235, 718, 536, 833))

    sim_auth = verifier.compute_similarity(crop, auth_path)
    sim_imp = verifier.compute_similarity(crop, imp_path)

    # Strictly verify requirement: Authentic > 0.85, Impostor < 0.55
    assert sim_auth >= 0.85, f"Authentic similarity ({sim_auth}) should be >= 0.85"
    assert sim_imp <= 0.55, f"Impostor similarity ({sim_imp}) should be <= 0.55"

    res_multi = verifier.verify_against_references(crop, [auth_path, imp_path])
    assert res_multi["decision"] == "GENUINE"
    assert res_multi["best_reference_index"] == 0
    assert len(res_multi["individual_scores"]) == 2
    assert res_multi["individual_scores"][0] > res_multi["individual_scores"][1]


def test_phase4_fallback_stroke_engine_accuracy_and_separation():
    """Verify fallback biometric stroke topology engine achieves > 0.85 authentic and < 0.55 impostor."""
    verifier = ProductionSignatureVerifier()
    verifier.model = None  # Force non-neural biometric engine fallback

    auth_path = str(BASE_DIR / "sample_test_run/reference_authentic.jpg")
    imp_path = str(BASE_DIR / "sample_test_run/reference_impostor.jpg")
    doc_path = str(BASE_DIR / "sample_test_run/sample_document.jpg")

    doc_img = Image.open(doc_path)
    crop = doc_img.crop((235, 718, 536, 833))

    sim_auth = verifier.compute_similarity(crop, auth_path)
    sim_imp = verifier.compute_similarity(crop, imp_path)

    assert sim_auth >= 0.85, f"Fallback authentic ({sim_auth}) should be >= 0.85"
    assert sim_imp <= 0.55, f"Fallback impostor ({sim_imp}) should be <= 0.55"


def test_phase2_mode2_reference_document_detection_and_verification():
    """Verify Mode 2 reference upload: Full document passed as reference signature."""
    from mlops.pipeline import UnifiedSignaturePipeline
    pipeline = UnifiedSignaturePipeline()

    doc_path = str(BASE_DIR / "sample_test_run/sample_document.jpg")
    # Use full document as reference
    output = pipeline.process_document(
        document_path=doc_path,
        reference_paths=[doc_path],
        save_crops_dir=str(BASE_DIR / "sample_test_run/crops")
    )
    assert output["detected_signatures_count"] >= 1
    first_res = output["results"][0]
    assert first_res["decision"] == "GENUINE"
    assert first_res["similarity_score"] >= 0.85


def test_phase2_pdf_document_input_support(tmp_path):
    """Verify Phase 2 A: Document input supports PDF files."""
    from mlops.pipeline import UnifiedSignaturePipeline
    pipeline = UnifiedSignaturePipeline()

    doc_path = str(BASE_DIR / "sample_test_run/sample_document.jpg")
    auth_path = str(BASE_DIR / "sample_test_run/reference_authentic.jpg")

    # Generate synthetic PDF containing document image
    pdf_path = str(tmp_path / "test_document.pdf")
    img = Image.open(doc_path)
    img.save(pdf_path, "PDF")

    output = pipeline.process_document(
        document_path=pdf_path,
        reference_paths=[auth_path],
        save_crops_dir=str(tmp_path / "crops")
    )
    assert output["detected_signatures_count"] >= 1
    first_res = output["results"][0]
    assert first_res["decision"] == "GENUINE"
    assert first_res["similarity_score"] >= 0.85


def test_phases_5_6_7_end_to_end_job_submission_and_inspection():
    """Test full API lifecycle: submit sync job, verify DB persistence, inspect crop URLs."""
    doc_path = str(BASE_DIR / "sample_test_run/sample_document.jpg")
    auth_path = str(BASE_DIR / "sample_test_run/reference_authentic.jpg")
    imp_path = str(BASE_DIR / "sample_test_run/reference_impostor.jpg")

    with open(doc_path, "rb") as f_doc, open(auth_path, "rb") as f_auth, open(imp_path, "rb") as f_imp:
        doc_upload = UploadFile(filename="sample_document.jpg", file=f_doc)
        auth_upload = UploadFile(filename="reference_authentic.jpg", file=f_auth)
        imp_upload = UploadFile(filename="reference_impostor.jpg", file=f_imp)

        db = SessionLocal()
        try:
            res = verify_signature_endpoint(
                document=doc_upload,
                references=[auth_upload, imp_upload],
                sync=True,
                db=db
            )
        finally:
            db.close()

    assert res.status == "COMPLETED"
    assert res.detected_count >= 1
    assert res.document_url and res.document_url.startswith("/uploads/")

    results = res.results
    assert len(results) >= 1
    first_res = results[0]
    assert first_res.decision == "GENUINE"
    assert first_res.similarity_score >= 0.85
    assert first_res.crop_url is not None
    assert first_res.crop_url.startswith("/uploads/"), f"Crop URL should start with /uploads/: {first_res.crop_url}"

    # Verify crop URL maps to an actual readable image file on disk
    rel_crop = first_res.crop_url.replace("/uploads/", "")
    crop_fs_path = os.path.join(settings.UPLOAD_DIR, rel_crop)
    assert os.path.exists(crop_fs_path), f"Crop image file does not exist on disk: {crop_fs_path}"
    with Image.open(crop_fs_path) as crop_img:
        assert crop_img.width > 20 and crop_img.height > 10

    # Inspect job via get_job_status (Phase 7 audit inspection)
    db = SessionLocal()
    try:
        inspect_data = get_job_status(job_id=res.job_id, db=db)
    finally:
        db.close()

    assert inspect_data.job_id == res.job_id
    assert inspect_data.status == "COMPLETED"
    assert inspect_data.document_url.startswith("/uploads/")
    assert inspect_data.results[0].crop_url.startswith("/uploads/")
