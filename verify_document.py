import os
import sys
import argparse
import json
from PIL import Image

from ml.detector import RTDETRSignatureDetector
from ml.verification import ResNet50SignatureVerifier
from ml.engine import SignatureEngine

def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Signature Detection & Verification Runner"
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="Path to full document image (scanned invoice, contract, or form)"
    )
    parser.add_argument(
        "--ref",
        type=str,
        required=True,
        nargs="+",
        help="Path to one or more authentic reference signature images"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.82,
        help="Calibrated cosine similarity verification threshold (default: 0.82)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)"
    )
    args = parser.parse_args()

    # 1. Validation
    if not os.path.exists(args.doc):
        print(f"[!] Error: Document image not found: {args.doc}")
        sys.exit(1)

    for r in args.ref:
        if not os.path.exists(r):
            print(f"[!] Error: Reference signature image not found: {r}")
            sys.exit(1)

    print("======================================================================")
    print("🖋️  SIGNATURE DETECTION & VERIFICATION END-TO-END PIPELINE")
    print("======================================================================")
    print(f"[*] Document Input:            {args.doc}")
    print(f"[*] Reference Signature(s):    {args.ref}")
    print(f"[*] Calibrated Threshold:       tau = {args.threshold}")
    print("----------------------------------------------------------------------")

    # 2. Initialize Models
    print("[*] Loading Apache 2.0 Detection Model (RT-DETR-R50)...")
    detector = RTDETRSignatureDetector()

    print("[*] Loading Verification Embedding Model (ResNet-50, 2048-dim)...")
    verifier = ResNet50SignatureVerifier(pretrained=True)

    print("[*] Assembling unified SignatureEngine...")
    engine = SignatureEngine(
        detector=detector,
        verifier=verifier,
        verification_threshold=args.threshold
    )

    # 3. Load Images
    doc_image = Image.open(args.doc).convert("RGB")
    ref_images = [Image.open(r).convert("RGB") for r in args.ref]

    # 4. Execute Full Pipeline:
    # Document -> Detect -> Bounding Box -> Crop -> Preprocess -> ResNet Embedding -> Cosine Sim -> Decision
    print("\n[*] Processing Document...")
    result = engine.process_document(
        document_image=doc_image,
        reference_images=ref_images,
        detection_confidence=args.conf
    )

    # 5. Formatted Output for Terminal & Teacher Presentation
    print("======================================================================")
    print(f"📊 PIPELINE EXECUTION RESULTS")
    print("======================================================================")
    print(f"• Document Status:             {result['status']}")
    print(f"• Final Verification Decision: {result.get('decision', 'N/A')}")
    print(f"• Signatures Detected in Doc:  {result.get('detections_count', 0)}")
    print(f"• Processing Latency:          {result['processing_time_sec']} seconds")
    print("----------------------------------------------------------------------")

    matches = result.get("matches", [])
    if not matches:
        print("[!] No candidate signatures detected on the provided document.")
    else:
        for m in matches:
            idx = m["candidate_index"]
            box = m["box_xyxy"]
            det_conf = m["detection_confidence"]
            sim_score = m["similarity_score"]
            is_match = m["is_match"]
            decision_label = m["decision"]

            print(f"Candidate #{idx + 1}:")
            print(f"  - Bounding Box [xmin, ymin, xmax, ymax]: {box}")
            print(f"  - Detector Confidence:                 {det_conf * 100:.1f}%")
            print(f"  - Cosine Similarity vs. Reference:     {sim_score:.4f}")
            print(f"  - Match Evaluation (tau >= {args.threshold}):          {'✅ MATCH (Genuine)' if is_match else '❌ NO MATCH (Suspect / Forgery)'}")

    print("======================================================================")

if __name__ == "__main__":
    main()
