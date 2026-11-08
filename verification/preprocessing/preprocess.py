"""
Signature Preprocessing & Normalization Pipeline for Verification
Applies grayscale conversion, adaptive contrast enhancement, aspect-ratio-preserving padding,
and standard tensor normalization for Siamese and Triplet metric learning models.
"""

import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms as T


def preprocess_signature_image(
    image: Image.Image,
    target_size: tuple = (224, 224),
    enhance_contrast: bool = True,
    to_grayscale: bool = True
) -> Image.Image:
    """
    Standardizes a signature crop for embedding generation.
    
    Args:
        image: Source cropped PIL image.
        target_size: Output (width, height) dimensions.
        enhance_contrast: Whether to apply autocontrast to emphasize ink strokes.
        to_grayscale: Whether to convert image to single-channel luminance.
        
    Returns:
        Standardized PIL Image with aspect-ratio-preserving letterboxing.
    """
    # 1. Convert to RGB or L
    if to_grayscale:
        img = image.convert("L")
        if enhance_contrast:
            img = ImageOps.autocontrast(img, cutoff=2)
        # Convert back to 3-channel for standard CNN backbones (ResNet, VGG, MobileNet)
        img = img.convert("RGB")
    else:
        img = image.convert("RGB")
        if enhance_contrast:
            img = ImageOps.autocontrast(img, cutoff=2)

    # 2. Aspect-ratio-preserving letterbox pad (do not distort signature stroke aspect ratio)
    target_w, target_h = target_size
    img_w, img_h = img.size

    scale = min(target_w / img_w, target_h / img_h)
    new_w = max(1, int(img_w * scale))
    new_h = max(1, int(img_h * scale))

    resized_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Create white canvas (background for documents)
    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized_img, (paste_x, paste_y))

    return canvas


def signature_to_tensor(
    image: Image.Image,
    target_size: tuple = (224, 224)
) -> torch.Tensor:
    """
    Transforms PIL image to normalized (3, H, W) PyTorch tensor suitable for model inference.
    """
    processed = preprocess_signature_image(image, target_size=target_size)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(processed)
