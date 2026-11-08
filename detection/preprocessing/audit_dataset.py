"""
Dataset Audit & Integrity Validator for Signature Detection Dataset
Analyzes YOLO-formatted bounding boxes, validates normalized coordinate ranges [0, 1],
identifies true negative (empty) background images, and computes statistical summaries.
"""

import os
import glob
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image


def audit_split(dataset_root: str, split_name: str) -> Dict[str, Any]:
    images_dir = os.path.join(dataset_root, split_name, "images")
    labels_dir = os.path.join(dataset_root, split_name, "labels")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    image_paths = glob.glob(os.path.join(images_dir, "*.*"))
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]

    stats = {
        "split": split_name,
        "total_images": len(image_paths),
        "total_labels": 0,
        "annotated_images": 0,
        "empty_images": 0,
        "total_bounding_boxes": 0,
        "classes": set(),
        "invalid_coords_count": 0,
        "resolutions": set(),
    }

    sample_check_limit = 200
    checked_samples = 0

    for img_path in image_paths:
        base_name = Path(img_path).stem
        label_path = os.path.join(labels_dir, f"{base_name}.txt")

        # Check resolution on a subset to confirm standard resolution
        if checked_samples < sample_check_limit:
            with Image.open(img_path) as img:
                stats["resolutions"].add(img.size)
            checked_samples += 1

        if not os.path.exists(label_path):
            stats["empty_images"] += 1
            continue

        stats["total_labels"] += 1
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if not lines:
            stats["empty_images"] += 1
            continue

        stats["annotated_images"] += 1
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, width, height = parts
            stats["classes"].add(class_id)
            stats["total_bounding_boxes"] += 1

            # Validate normalized coordinate bounds
            try:
                xc, yc, w, h = float(x_center), float(y_center), float(width), float(height)
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    stats["invalid_coords_count"] += 1
            except ValueError:
                stats["invalid_coords_count"] += 1

    return stats


def run_full_dataset_audit(dataset_root: str = "merged_signature_dataset") -> Dict[str, Any]:
    print(f"[*] Starting full dataset audit on: {os.path.abspath(dataset_root)}")
    results = {}
    splits = ["train", "valid", "test"]

    for split in splits:
        split_path = os.path.join(dataset_root, split)
        if os.path.exists(split_path):
            print(f"[*] Auditing split: {split}...")
            stats = audit_split(dataset_root, split)
            results[split] = stats
            print(f"    [+] {split}: {stats['total_images']} images | "
                  f"{stats['total_bounding_boxes']} boxes | "
                  f"{stats['empty_images']} true-negative blanks | "
                  f"classes: {stats['classes']}")

    return results


if __name__ == "__main__":
    audit = run_full_dataset_audit()
    print("\n[✓] Audit successfully completed.")
