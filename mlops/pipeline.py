"""
Unified End-to-End Signature Pipeline
Connects Document Input -> Signature Detection -> Adaptive Cropping -> Metric Verification.
"""

import os
from typing import List, Dict, Any, Union
from PIL import Image

from detection.selected_model.detector import ProductionSignatureDetector
from verification.selected_model.verifier import ProductionSignatureVerifier
from verification.preprocessing.signature_cropper import crop_signature


class UnifiedSignaturePipeline:
    """
    End-to-End Pipeline executing two-stage inference:
    Stage 1: Detect signature bounding boxes on document.
    Stage 2: Crop regions and compare against reference signatures.
    """
    def __init__(self):
        self.detector = ProductionSignatureDetector()
        self.verifier = ProductionSignatureVerifier()

    def process_document(
        self,
        document_path: str,
        reference_paths: List[str],
        save_crops_dir: str = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end detection and verification.
        """
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Document not found: {document_path}")

        doc_img = Image.open(document_path).convert("RGB")
        detections = self.detector.detect(doc_img)

        results = []
        if save_crops_dir:
            os.makedirs(save_crops_dir, exist_ok=True)

        for idx, det in enumerate(detections):
            bbox = det["bbox"]
            confidence = det["confidence"]
            
            # Stage 1.5: Adaptive signature cropping
            cropped = crop_signature(doc_img, bbox, padding_ratio=0.05, coord_format="xyxy")
            
            crop_path = None
            if save_crops_dir:
                crop_filename = f"crop_sig_{idx}.jpg"
                crop_path = os.path.join(save_crops_dir, crop_filename)
                cropped.save(crop_path, "JPEG")

            # Stage 2: Biometric verification against references
            verif = self.verifier.verify_against_references(cropped, reference_paths)

            results.append({
                "signature_index": idx,
                "bbox": bbox,
                "confidence_detection": round(confidence, 4),
                "similarity_score": verif["aggregated_similarity"],
                "decision": verif["decision"],
                "individual_scores": verif["individual_scores"],
                "best_reference": reference_paths[verif["best_reference_index"]],
                "crop_path": crop_path
            })

        return {
            "document_path": document_path,
            "detected_signatures_count": len(results),
            "results": results
        }
