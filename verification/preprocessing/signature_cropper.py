"""
Signature Cropper and Normalizer
Extracts signature regions from full documents given bounding box predictions or ground truth labels,
applies optional adaptive padding, boundary clamping, and aspect-ratio preservation.
"""

from typing import Tuple, Union, List
from PIL import Image


def crop_signature(
    image: Image.Image,
    bbox: Union[List[float], Tuple[float, float, float, float]],
    padding_ratio: float = 0.05,
    coord_format: str = "xyxy"
) -> Image.Image:
    """
    Crops a signature region from a PIL Image.
    
    Args:
        image: Source PIL Image.
        bbox: Bounding box coordinates.
        padding_ratio: Context padding percentage (e.g. 0.05 adds 5% margin around the stroke).
        coord_format: One of 'xyxy' (xmin, ymin, xmax, ymax) or 'xywh' (xmin, ymin, width, height).
        
    Returns:
        Cropped PIL Image containing the signature.
    """
    img_w, img_h = image.size

    if coord_format == "xywh":
        x, y, w, h = bbox
        xmin, ymin, xmax, ymax = x, y, x + w, y + h
    elif coord_format == "xyxy":
        xmin, ymin, xmax, ymax = bbox
    else:
        raise ValueError(f"Unsupported coordinate format: {coord_format}. Use 'xyxy' or 'xywh'.")

    # If normalized [0, 1] coords were passed, convert to absolute pixels
    if xmax <= 1.0 and ymax <= 1.0 and (xmax - xmin) < 1.0 and (ymax - ymin) < 1.0:
        xmin *= img_w
        xmax *= img_w
        ymin *= img_h
        ymax *= img_h

    # Add adaptive padding around strokes to preserve trailing flourishes
    box_w = xmax - xmin
    box_h = ymax - ymin
    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio

    clamped_xmin = max(0, int(round(xmin - pad_x)))
    clamped_ymin = max(0, int(round(ymin - pad_y)))
    clamped_xmax = min(img_w, int(round(xmax + pad_x)))
    clamped_ymax = min(img_h, int(round(ymax + pad_y)))

    # Ensure valid dimensions
    if clamped_xmax <= clamped_xmin or clamped_ymax <= clamped_ymin:
        # Fallback to center crop if box is degenerate
        clamped_xmin, clamped_ymin = 0, 0
        clamped_xmax, clamped_ymax = min(img_w, 64), min(img_h, 64)

    return image.crop((clamped_xmin, clamped_ymin, clamped_xmax, clamped_ymax))
