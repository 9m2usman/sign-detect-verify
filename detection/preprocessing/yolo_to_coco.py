"""
YOLO to COCO Annotation Converter
Converts YOLO-formatted text annotations (normalized class, x_center, y_center, width, height)
into standard COCO 1.0 JSON format for RTMDet (MMDetection / MMYOLO) and standard vision frameworks.
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image


def convert_split_yolo_to_coco(
    images_dir: str,
    labels_dir: str,
    output_json_path: str,
    category_id: int = 1,
    category_name: str = "signature",
    image_width: int = 640,
    image_height: int = 640,
    skip_empty: bool = False
) -> Dict[str, Any]:
    """
    Converts a single split from YOLO txt format to COCO json.
    """
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    coco_data = {
        "info": {
            "description": "Signature Object Detection Dataset (COCO Format)",
            "version": "1.0",
            "year": 2026,
            "contributor": "Signature Detection & MLOps Platform"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Apache-2.0 Compatible Research License",
                "url": ""
            }
        ],
        "categories": [
            {
                "id": category_id,
                "name": category_name,
                "supercategory": "document_element"
            }
        ],
        "images": [],
        "annotations": []
    }

    annotation_id = 1
    image_id = 1

    for img_path in image_files:
        base_name = Path(img_path).stem
        label_file = os.path.join(labels_dir, f"{base_name}.txt")

        # Check if label exists
        has_annotations = False
        boxes = []
        if os.path.exists(label_file):
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        boxes.append([float(p) for p in parts])

        if skip_empty and not boxes:
            continue

        file_name = os.path.basename(img_path)
        coco_data["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": image_width,
            "height": image_height,
            "license": 1
        })

        for box in boxes:
            cls_id, xc, yc, w, h = box
            # Convert normalized YOLO (cx, cy, w, h) to absolute COCO (x_min, y_min, w, h)
            abs_w = w * image_width
            abs_h = h * image_height
            abs_x = (xc * image_width) - (abs_w / 2.0)
            abs_y = (yc * image_height) - (abs_h / 2.0)

            # Clamp boundaries
            abs_x = max(0.0, min(float(image_width - 1), abs_x))
            abs_y = max(0.0, min(float(image_height - 1), abs_y))
            abs_w = max(1.0, min(float(image_width - abs_x), abs_w))
            abs_h = max(1.0, min(float(image_height - abs_y), abs_h))

            area = abs_w * abs_h

            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [round(abs_x, 2), round(abs_y, 2), round(abs_w, 2), round(abs_h, 2)],
                "area": round(area, 2),
                "iscrowd": 0,
                "segmentation": []
            })
            annotation_id += 1

        image_id += 1

    # Save output
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)

    return {
        "output_path": output_json_path,
        "images_count": len(coco_data["images"]),
        "annotations_count": len(coco_data["annotations"])
    }
