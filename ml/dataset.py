import os
import glob
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class BoundingBox:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    confidence: Optional[float] = None

    def to_xyxy(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized YOLO cx, cy, w, h to absolute pixel coordinates (xmin, ymin, xmax, ymax)."""
        xmin = max(0, int((self.cx - self.w / 2.0) * img_width))
        ymin = max(0, int((self.cy - self.h / 2.0) * img_height))
        xmax = min(img_width, int((self.cx + self.w / 2.0) * img_width))
        ymax = min(img_height, int((self.cy + self.h / 2.0) * img_height))
        return xmin, ymin, xmax, ymax

    def to_coco(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Convert normalized YOLO to COCO absolute [xmin, ymin, width, height]."""
        xmin = max(0.0, (self.cx - self.w / 2.0) * img_width)
        ymin = max(0.0, (self.cy - self.h / 2.0) * img_height)
        width = min(img_width - xmin, self.w * img_width)
        height = min(img_height - ymin, self.h * img_height)
        return xmin, ymin, width, height


class YOLODatasetParser:
    """
    Parser for YOLO-formatted signature datasets.
    Parses images and corresponding label text files without modifying raw assets.
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")

    def get_split_samples(self, split: str = "train") -> List[Dict[str, Any]]:
        """
        Retrieves all valid image-annotation pairs for a given split ('train', 'valid', 'test').
        Returns list of dicts: {'image_path': str, 'label_path': str, 'boxes': List[BoundingBox]}
        """
        img_dir = os.path.join(self.root_dir, split, "images")
        lbl_dir = os.path.join(self.root_dir, split, "labels")

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            raise FileNotFoundError(f"Split folders for '{split}' missing in {self.root_dir}")

        image_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        samples = []

        for img_path in image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")

            boxes = []
            if os.path.exists(lbl_path):
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            conf = float(parts[5]) if len(parts) > 5 else None
                            boxes.append(BoundingBox(cls_id, cx, cy, w, h, conf))

            samples.append({
                "image_path": img_path,
                "label_path": lbl_path if os.path.exists(lbl_path) else None,
                "boxes": boxes,
                "has_signature": len(boxes) > 0
            })

        return samples

    def summarize(self) -> Dict[str, Any]:
        """Provides verification summary statistics across train, valid, test splits."""
        stats = {}
        for split in ["train", "valid", "test"]:
            samples = self.get_split_samples(split)
            total_images = len(samples)
            total_boxes = sum(len(s["boxes"]) for s in samples)
            negatives = sum(1 for s in samples if not s["has_signature"])
            stats[split] = {
                "total_images": total_images,
                "total_boxes": total_boxes,
                "negatives": negatives,
                "positives": total_images - negatives
            }
        return stats
