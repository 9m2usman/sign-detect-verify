#!/usr/bin/env python3
"""
CLI Tool: End-to-End Signature Detection & Verification Runner
License: Apache 2.0 / BSD-3-Clause (Zero AGPL)
"""

import os
import sys
import argparse
import json
from mlops.pipeline import UnifiedSignaturePipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify document signatures against authentic reference signatures."
    )
    parser.add_argument(
        "--doc", "--document",
        dest="doc_path",
        required=True,
        help="Path to the document image file (JPG, PNG)."
    )
    parser.add_argument(
        "--ref", "--reference",
        dest="ref_paths",
        nargs="+",
        required=True,
        help="One or more paths to genuine reference signature images."
    )
    parser.add_argument(
        "--crops-dir",
        default="sample_test_run/extracted_crops",
        help="Directory to save cropped signature candidates."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of human-readable text."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.doc_path):
        print(f"[Error] Document path does not exist: {args.doc_path}", file=sys.stderr)
        sys.exit(1)

    for ref in args.ref_paths:
        if not os.path.exists(ref):
            print(f"[Error] Reference path does not exist: {ref}", file=sys.stderr)
            sys.exit(1)

    pipeline = UnifiedSignaturePipeline()
    outcome = pipeline.process_document(
        document_path=args.doc_path,
        reference_paths=args.ref_paths,
        save_crops_dir=args.crops_dir
    )

    if args.json:
        print(json.dumps(outcome, indent=2))
        return

    print("=" * 65)
    print(" 🖋️  SIGNATURE DETECTION & VERIFICATION REPORT")
    print("=" * 65)
    print(f"Document: {outcome['document_path']}")
    print(f"Detections Found: {outcome['detected_signatures_count']}")
    print("-" * 65)

    for res in outcome["results"]:
        status_symbol = "✅" if res["decision"] == "GENUINE" else ("❌" if res["decision"] == "FORGED" else "⚠️")
        print(f"Signature #{res['signature_index'] + 1}:")
        print(f"  • Bounding Box       : {res['bbox']}")
        print(f"  • Detection Conf.    : {res['confidence_detection']}")
        print(f"  • Similarity Score   : {res['similarity_score']:.4f}")
        print(f"  • Decision           : {status_symbol} {res['decision']}")
        print(f"  • Best Reference     : {res['best_reference']}")
        if res.get("crop_path"):
            print(f"  • Extracted Crop     : {res['crop_path']}")
        print("-" * 65)


if __name__ == "__main__":
    main()
