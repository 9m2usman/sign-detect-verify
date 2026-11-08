"""
End-to-End Integration Tests for Signature Detection & Verification Pipeline.
Tests document ingestion, signature bounding box localization, adaptive cropping,
feature embedding generation, cosine similarity calculation, and decision thresholding.
"""

import os
import pytest
from PIL import Image, ImageDraw
from mlops.pipeline import UnifiedSignaturePipeline


@pytest.fixture
def sample_test_assets(tmp_path):
    """Creates synthetic document with a signature and reference signatures."""
    doc_path = str(tmp_path / "test_doc.jpg")
    ref_auth_path = str(tmp_path / "test_ref_auth.jpg")
    ref_forg_path = str(tmp_path / "test_ref_forg.jpg")

    # Document (640x640 with black text and signature flourish in lower right)
    doc_img = Image.new("RGB", (640, 640), (255, 255, 255))
    draw = ImageDraw.Draw(doc_img)
    draw.text((50, 50), "Standard Loan Agreement Document", fill=(0, 0, 0))
    # Draw signature flourish in lower right
    draw.arc((400, 480, 550, 560), 0, 360, fill=(0, 0, 128), width=3)
    doc_img.save(doc_path, "JPEG")

    # Reference authentic signature
    ref_auth = Image.new("RGB", (300, 150), (255, 255, 255))
    d_auth = ImageDraw.Draw(ref_auth)
    d_auth.arc((50, 30, 250, 120), 0, 360, fill=(0, 0, 128), width=3)
    ref_auth.save(ref_auth_path, "JPEG")

    # Reference different signature (impostor)
    ref_forg = Image.new("RGB", (300, 150), (255, 255, 255))
    d_forg = ImageDraw.Draw(ref_forg)
    d_forg.line((20, 20, 280, 130), fill=(255, 0, 0), width=4)
    ref_forg.save(ref_forg_path, "JPEG")

    return {
        "doc": doc_path,
        "ref_auth": ref_auth_path,
        "ref_forg": ref_forg_path,
        "tmp_dir": str(tmp_path)
    }


def test_unified_pipeline_end_to_end(sample_test_assets):
    pipeline = UnifiedSignaturePipeline()
    
    crops_output_dir = os.path.join(sample_test_assets["tmp_dir"], "crops")
    output = pipeline.process_document(
        document_path=sample_test_assets["doc"],
        reference_paths=[sample_test_assets["ref_auth"], sample_test_assets["ref_forg"]],
        save_crops_dir=crops_output_dir
    )

    assert output["detected_signatures_count"] >= 1
    assert len(output["results"]) >= 1

    first_result = output["results"][0]
    assert "bbox" in first_result
    assert len(first_result["bbox"]) == 4
    assert "similarity_score" in first_result
    assert -1.0 <= first_result["similarity_score"] <= 1.0
    assert first_result["decision"] in ["GENUINE", "FORGED", "REVIEW"]
    assert len(first_result["individual_scores"]) == 2
    assert os.path.exists(first_result["crop_path"])
