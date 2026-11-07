import os
from ml.dataset import YOLODatasetParser, BoundingBox
from ml.utils import extract_signature_crops, preprocess_signature_for_verification, compute_iou
from PIL import Image
import numpy as np

def test_bounding_box_conversion():
    # cx=0.5, cy=0.5, w=0.2, h=0.2 on 1000x1000 image
    bbox = BoundingBox(class_id=0, cx=0.5, cy=0.5, w=0.2, h=0.2)
    xmin, ymin, xmax, ymax = bbox.to_xyxy(1000, 1000)
    assert xmin == 400
    assert ymin == 400
    assert xmax == 600
    assert ymax == 600

def test_compute_iou():
    boxA = (100, 100, 200, 200)
    boxB = (100, 100, 200, 200)
    assert abs(compute_iou(boxA, boxB) - 1.0) < 0.01

    boxC = (300, 300, 400, 400)
    assert compute_iou(boxA, boxC) == 0.0


def test_extract_signature_crops():
    # Create white image with black box
    img = Image.new("RGB", (640, 640), color="white")
    boxes = [(100, 100, 200, 200)]
    crops = extract_signature_crops(img, boxes, padding_ratio=0.1)
    assert len(crops) == 1
    # Check that padded size is greater than original 100x100
    cw, ch = crops[0].size
    assert cw >= 100
    assert ch >= 100

def test_preprocess_signature_normalization():
    img = Image.new("RGB", (300, 150), color="white")
    preprocessed = preprocess_signature_for_verification(img, target_size=(224, 224))
    assert preprocessed.size == (224, 224)
    assert preprocessed.mode == "RGB"

def test_dataset_parser_local_split():
    dataset_path = "merged_signature_dataset"
    if os.path.exists(dataset_path):
        parser = YOLODatasetParser(dataset_path)
        val_samples = parser.get_split_samples("valid")
        assert len(val_samples) == 3287
        assert val_samples[0]["has_signature"] in [True, False]
