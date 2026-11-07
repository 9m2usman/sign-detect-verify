import time
from typing import List, Dict, Any, Optional
from PIL import Image

from ml.detector import BaseSignatureDetector
from ml.verification import BaseSignatureEmbeddingModel
from ml.utils import extract_signature_crops, preprocess_signature_for_verification

class SignatureEngine:
    """
    Unified Production Signature Detection & Verification Pipeline.
    Integrates Stage A (Detection) and Stage B (Verification) with evidence-based decisions.
    """
    def __init__(
        self,
        detector: BaseSignatureDetector,
        verifier: BaseSignatureEmbeddingModel,
        verification_threshold: float = 0.82
    ):
        self.detector = detector
        self.verifier = verifier
        self.verification_threshold = verification_threshold

    def process_document(
        self,
        document_image: Image.Image,
        reference_images: List[Image.Image],
        detection_confidence: float = 0.40
    ) -> Dict[str, Any]:
        """
        Executes end-to-end detection and verification:
        1. Validates input images.
        2. Detects signatures on the document.
        3. Crops each candidate signature.
        4. Preprocesses crops and reference signatures.
        5. Computes cosine similarity against provided reference(s).
        6. Applies calibrated threshold decision.
        """
        start_time = time.time()

        # 1. Signature Detection
        detections = self.detector.detect(document_image, confidence_threshold=detection_confidence)
        
        if not detections:
            return {
                "status": "NO_SIGNATURE_DETECTED",
                "message": "No signature candidates detected on the document at the given confidence threshold.",
                "detections": [],
                "matches": [],
                "processing_time_sec": round(time.time() - start_time, 4)
            }

        # 2. Extract bounding boxes
        boxes_xyxy = [d["box_xyxy"] for d in detections]
        signature_crops = extract_signature_crops(document_image, boxes_xyxy)

        # 3. Process reference embeddings
        ref_embeddings = []
        for ref_img in reference_images:
            preprocessed_ref = preprocess_signature_for_verification(ref_img)
            ref_emb = self.verifier.extract_embedding(preprocessed_ref)
            ref_embeddings.append(ref_emb)

        # 4. Compare each detected signature against references
        matches = []
        for idx, (det, crop) in enumerate(zip(detections, signature_crops)):
            preprocessed_crop = preprocess_signature_for_verification(crop)
            query_emb = self.verifier.extract_embedding(preprocessed_crop)

            # Compute similarities against all references (Aggregation Strategy: Max Similarity)
            similarities = [self.verifier.compute_similarity(query_emb, ref_emb) for ref_emb in ref_embeddings]
            best_score = max(similarities) if similarities else 0.0
            best_ref_idx = similarities.index(best_score) if similarities else None
            
            is_verified = best_score >= self.verification_threshold

            matches.append({
                "candidate_index": idx,
                "box_xyxy": det["box_xyxy"],
                "detection_confidence": det["confidence"],
                "similarity_score": best_score,
                "matched_reference_index": best_ref_idx,
                "is_match": is_verified,
                "decision": "GENUINE_MATCH" if is_verified else "SUSPECT_OR_FORGERY"
            })

        # Overall document verification decision
        overall_match = any(m["is_match"] for m in matches)

        return {
            "status": "COMPLETED",
            "decision": "VERIFIED" if overall_match else "FAILED_VERIFICATION",
            "detector_architecture": self.detector.get_model_name(),
            "embedding_dimension": self.verifier.get_embedding_dim(),
            "calibrated_threshold": self.verification_threshold,
            "detections_count": len(detections),
            "matches": matches,
            "processing_time_sec": round(time.time() - start_time, 4)
        }
