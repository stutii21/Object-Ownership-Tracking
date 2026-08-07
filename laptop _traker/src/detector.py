"""
Detector module.

Wraps a pretrained Ultralytics YOLO model, restricted to the classes we care
about (person, laptop). No custom training is required for these two classes
since both already exist in COCO, which the pretrained weights were trained on.

If you later add a class that ISN'T in COCO (e.g. a specific badge, a
custom conveyor part), you will need to fine-tune on your own annotated
data (see README -> "Custom classes" section) and point MODEL_WEIGHTS at
your fine-tuned .pt file instead.
"""
from __future__ import annotations

import numpy as np
from ultralytics import YOLO

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class Detector:
    def __init__(self, weights: str = config.MODEL_WEIGHTS,
                 classes: list[int] = config.DETECT_CLASSES,
                 conf: float = config.CONF_THRESHOLD,
                 iou: float = config.IOU_THRESHOLD_NMS,
                 imgsz: int = config.INFERENCE_IMGSZ):
        self.model = YOLO(weights)
        self.classes = classes
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray):
        """
        Run detection on a single BGR frame (as read by OpenCV).

        Returns:
            boxes: (N, 4) float array, xyxy pixel coords
            confidences: (N,) float array
            class_ids: (N,) int array (COCO ids)
        """
        results = self.model.predict(
            source=frame,
            classes=self.classes,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return (np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int))

        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        return boxes, confidences, class_ids
