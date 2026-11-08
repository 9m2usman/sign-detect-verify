"""
Unit Tests for Preprocessing Pipeline
Tests bounding box cropping, clamping, aspect-ratio letterboxing, and tensor transformation.
"""

import pytest
from PIL import Image
import torch

from verification.preprocessing.signature_cropper import crop_signature
from verification.preprocessing.preprocess import preprocess_signature_image, signature_to_tensor


def test_crop_signature_xyxy():
    # Create synthetic 640x640 test image
    img = Image.new("RGB", (640, 640), (255, 255, 255))
    bbox = [100, 150, 300, 250]  # xmin, ymin, xmax, ymax
    cropped = crop_signature(img, bbox, padding_ratio=0.0, coord_format="xyxy")
    
    assert cropped.size == (200, 100)


def test_crop_signature_xywh():
    img = Image.new("RGB", (640, 640), (255, 255, 255))
    bbox = [50, 50, 100, 80]  # xmin, ymin, width, height
    cropped = crop_signature(img, bbox, padding_ratio=0.0, coord_format="xywh")
    
    assert cropped.size == (100, 80)


def test_crop_signature_boundary_clamping():
    # Bbox extending beyond image boundaries
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    bbox = [180, 180, 250, 250]
    cropped = crop_signature(img, bbox, padding_ratio=0.1, coord_format="xyxy")
    
    assert cropped.size[0] <= 200
    assert cropped.size[1] <= 200


def test_preprocess_letterbox():
    # Test letterboxing preserves target dimensions
    img = Image.new("RGB", (400, 100), (0, 0, 0))  # Wide aspect ratio
    processed = preprocess_signature_image(img, target_size=(224, 224))
    
    assert processed.size == (224, 224)


def test_signature_to_tensor():
    img = Image.new("RGB", (300, 150), (128, 128, 128))
    tensor = signature_to_tensor(img, target_size=(224, 224))
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
