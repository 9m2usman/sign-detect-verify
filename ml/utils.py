import os
import json
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import numpy as np

def extract_signature_crops(
    image: Image.Image,
    bounding_boxes: List[Tuple[int, int, int, int]],
    padding_ratio: float = 0.05
) -> List[Image.Image]:
    """
    Crops signature regions from a document PIL Image given [xmin, ymin, xmax, ymax] boxes.
    Adds a dynamic padding around the bounding box without going out of image boundaries.
    """
    img_w, img_h = image.size
    crops = []

    for (xmin, ymin, xmax, ymax) in bounding_boxes:
        bw = xmax - xmin
        bh = ymax - ymin
        
        pad_x = int(bw * padding_ratio)
        pad_y = int(bh * padding_ratio)

        c_xmin = max(0, xmin - pad_x)
        c_ymin = max(0, ymin - pad_y)
        c_xmax = min(img_w, xmax + pad_x)
        c_ymax = min(img_h, ymax + pad_y)

        if c_xmax > c_xmin and c_ymax > c_ymin:
            crop = image.crop((c_xmin, c_ymin, c_xmax, c_ymax))
            crops.append(crop)

    return crops


def preprocess_signature_for_verification(
    signature_crop: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    apply_background_suppression: bool = True
) -> Image.Image:
    """
    Normalizes a signature crop for embedding generation:
    1. Converts to Grayscale.
    2. Optional Otsu-like background suppression to emphasize pen strokes over paper texture.
    3. Resizes with aspect-ratio preserving padding to target_size.
    4. Converts back to 3-channel RGB for ResNet/ViT feature extractors.
    """
    gray = signature_crop.convert("L")

    if apply_background_suppression:
        np_img = np.array(gray)
        # Background suppression: invert if dark background, suppress light background
        # Document signatures are typically dark ink on light paper
        threshold_val = np.percentile(np_img, 75)
        # Normalize contrast
        np_norm = np.clip((np_img.astype(np.float32) / max(1.0, threshold_val)) * 255.0, 0, 255).astype(np.uint8)
        gray = Image.fromarray(np_norm)

    # Aspect-ratio preserving resize with white letterboxing
    w, h = gray.size
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized_stroke = gray.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Create white canvas (255)
    canvas = Image.new("L", target_size, color=255)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized_stroke, (paste_x, paste_y))

    # Convert to 3-channel image for standard vision backbones
    rgb_normalized = canvas.convert("RGB")
    return rgb_normalized


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union (IoU) between two [xmin, ymin, xmax, ymax] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-7)
    return iou
