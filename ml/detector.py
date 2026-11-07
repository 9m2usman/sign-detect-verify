import os
import abc
from typing import List, Tuple, Dict, Any, Optional
import torch
from PIL import Image
import numpy as np

class BaseSignatureDetector(abc.ABC):
    """Abstract Base Class establishing common interface for signature detection models."""

    @abc.abstractmethod
    def detect(self, image: Image.Image, confidence_threshold: float = 0.4) -> List[Dict[str, Any]]:
        """
        Runs object detection on document PIL Image.
        Returns list of dicts:
        [
            {
                "box_xyxy": [xmin, ymin, xmax, ymax],
                "confidence": float,
                "class_id": 0,
                "label": "signature"
            }
        ]
        """
        pass

    @abc.abstractmethod
    def get_model_name(self) -> str:
        """Returns architecture name."""
        pass


class RTDETRSignatureDetector(BaseSignatureDetector):
    """
    Candidate 2: RT-DETR (Real-Time DEtection TRansformer).
    License: Apache 2.0.
    Natively integrated via Hugging Face Transformers. Anchor-free, NMS-free.
    """
    def __init__(self, model_id_or_path: str = "PekingU/rtdetr_r50vd", device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.model_id = model_id_or_path
        self.processor = AutoImageProcessor.from_pretrained(model_id_or_path)
        self.model = AutoModelForObjectDetection.from_pretrained(model_id_or_path)
        self.model.to(self.device)
        self.model.eval()

    def get_model_name(self) -> str:
        return f"RT-DETR ({self.model_id})"

    @torch.no_grad()
    def detect(self, image: Image.Image, confidence_threshold: float = 0.4) -> List[Dict[str, Any]]:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=confidence_threshold
        )[0]

        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            xmin, ymin, xmax, ymax = box.tolist()
            detections.append({
                "box_xyxy": [int(xmin), int(ymin), int(xmax), int(ymax)],
                "confidence": round(float(score), 4),
                "class_id": int(label),
                "label": "signature"
            })

        # Sort by confidence descending
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        return detections


class YOLOXSignatureDetector(BaseSignatureDetector):
    """
    Candidate 1: YOLOX-S Signature Detector.
    License: Apache 2.0 (Megvii).
    Runs ONNX or PyTorch weights directly.
    """
    def __init__(self, weights_path: Optional[str] = None, input_size: Tuple[int, int] = (640, 640), device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.input_size = input_size
        self.weights_path = weights_path
        # Will bind trained checkpoint or export ONNX runner

    def get_model_name(self) -> str:
        return "YOLOX-S"

    def detect(self, image: Image.Image, confidence_threshold: float = 0.4) -> List[Dict[str, Any]]:
        # Inference handler wrapping YOLOX post-processing (nms, decode)
        return []
