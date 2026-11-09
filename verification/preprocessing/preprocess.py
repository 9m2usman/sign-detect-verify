"""
Signature Preprocessing & Normalization Pipeline for Verification
Applies grayscale conversion, adaptive contrast enhancement, aspect-ratio-preserving padding,
and standard tensor normalization for Siamese and Triplet metric learning models.
Supports raster images (JPEG, PNG, WEBP, TIFF) and digital/scanned PDF documents.
"""

import os
import io
from typing import Union
import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms as T


def load_document_image(source: Union[str, bytes, Image.Image]) -> Image.Image:
    """
    Loads an image or PDF document page as a PIL RGB Image.
    Handles:
    - PIL.Image.Image instances directly.
    - Raw image bytes or PDF byte streams.
    - Image file paths (JPEG, PNG, TIFF, WEBP, BMP, etc.).
    - Scanned or digital PDF files via pypdf (extracts page images) or pdf2image fallback.
    """
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    if isinstance(source, bytes):
        if source.startswith(b"%PDF"):
            return _load_pdf_from_stream(io.BytesIO(source))
        return Image.open(io.BytesIO(source)).convert("RGB")

    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")
        if source.lower().endswith(".pdf"):
            with open(source, "rb") as f:
                return _load_pdf_from_stream(f)
        return Image.open(source).convert("RGB")

    raise TypeError(f"Unsupported document source type: {type(source)}")


def _load_pdf_from_stream(stream) -> Image.Image:
    """
    Renders the first page of a PDF document to a high-resolution PIL RGB Image.
    Uses pypdfium2 for direct vector/raster rasterization, with pypdf and pdf2image fallback.
    """
    # 1. High-fidelity rendering with pypdfium2 (handles both vector and scanned PDFs)
    try:
        import pypdfium2
        if hasattr(stream, "seek"):
            stream.seek(0)
            raw_bytes = stream.read()
        else:
            raw_bytes = stream
        pdf = pypdfium2.PdfDocument(raw_bytes)
        if len(pdf) > 0:
            # Render page 1 at scale 2.0 (approx 150-200 DPI) for high signature clarity
            page = pdf[0]
            pil_image = page.render(scale=2.0).to_pil().convert("RGB")
            return pil_image
    except Exception:
        pass

    # 2. Extract embedded images with pypdf
    try:
        import pypdf
        if hasattr(stream, "seek"):
            stream.seek(0)
        reader = pypdf.PdfReader(stream)
        for page in reader.pages:
            if hasattr(page, "images") and len(page.images) > 0:
                best_img = None
                best_area = 0
                for img_obj in page.images:
                    try:
                        pil_img = Image.open(io.BytesIO(img_obj.data)).convert("RGB")
                        area = pil_img.width * pil_img.height
                        if area > best_area:
                            best_area = area
                            best_img = pil_img
                    except Exception:
                        continue
                if best_img is not None:
                    return best_img
    except Exception:
        pass

    # 3. Fallback to pdf2image
    try:
        from pdf2image import convert_from_bytes
        if hasattr(stream, "seek"):
            stream.seek(0)
            data = stream.read()
        else:
            data = stream
        images = convert_from_bytes(data, first_page=1, last_page=1)
        if images:
            return images[0].convert("RGB")
    except Exception:
        pass

    raise ValueError("Could not render image from PDF document. Please verify the PDF is valid.")


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
