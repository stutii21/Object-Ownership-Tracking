"""
RF-DETR detector module.

Drop-in replacement for src/detector.py's YOLO-based Detector, using Roboflow's
RF-DETR (a transformer-based real-time detector). Same `.detect(frame)` interface,
so main.py can swap between backends via config.DETECTOR_BACKEND without any
other code changes.

WHY RF-DETR (per the original project brief):
- Transformer-based global attention -> fewer duplicate/overlapping boxes than
  YOLO in dense scenes (this video has 6-10 people in frame at once).
- Generally more robust to partial occlusion, which matters when someone's
  laptop is partially blocked by another person leaning in front of it.

IMPORTANT — network requirement:
RF-DETR downloads its pretrained COCO weights from `storage.googleapis.com`
the first time a given variant is instantiated. If you're running this in a
locked-down sandbox that only allowlists a handful of domains (pypi/npm/github),
that download will fail with a 403 `host_not_allowed`. On a normal machine, or
in Colab, this works with no extra steps. If you must work around a similar
allowlist, download the .pth manually on an unrestricted machine and place it
at the path printed in the error message (~/.roboflow/models/), or point
`weights_path` below at your own local copy.

Variants (accuracy vs speed, all pretrained on COCO -> includes person + laptop):
    RFDETRNano   - fastest, least accurate
    RFDETRSmall
    RFDETRMedium
    RFDETRBase   - good default balance
    RFDETRLarge  - most accurate, slowest
"""
from __future__ import annotations

import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

_VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}


class RFDETRDetector:
    def __init__(self,
                 variant: str = config.RFDETR_VARIANT,
                 classes: list[int] = config.DETECT_CLASSES,
                 conf: float = config.CONF_THRESHOLD,
                 weights_path: str | None = None):
        import rfdetr as rfdetr_module

        cls_name = _VARIANTS.get(variant.lower())
        if cls_name is None:
            raise ValueError(f"Unknown RF-DETR variant '{variant}'. Choose from {list(_VARIANTS)}")
        model_cls = getattr(rfdetr_module, cls_name)

        kwargs = {}
        if weights_path is not None:
            kwargs["pretrain_weights"] = weights_path

        self.model = model_cls(**kwargs)
        self.conf = conf

        # IMPORTANT: don't assume RF-DETR's own class ids line up with
        # Ultralytics' COCO ids (config.COCO_PERSON_ID=0 / COCO_LAPTOP_ID=63).
        # RF-DETR ships its own COCO_CLASSES name table, and its own
        # changelog notes at least one release where output class ids
        # shifted. Look ids up by NAME instead, then remap to the pipeline's
        # ids so ownership.py/main.py stay identical across backends.
        from rfdetr.assets.coco_classes import COCO_CLASSES
        name_to_id = ({name: idx for idx, name in COCO_CLASSES.items()} if isinstance(COCO_CLASSES, dict)
                      else {name: idx for idx, name in enumerate(COCO_CLASSES)})
        try:
            rfdetr_person_id = name_to_id["person"]
            rfdetr_laptop_id = name_to_id["laptop"]
        except KeyError as e:
            raise RuntimeError(
                "Could not find 'person'/'laptop' by name in RF-DETR's COCO_CLASSES -- "
                "inspect rfdetr.assets.coco_classes.COCO_CLASSES directly and update this mapping."
            ) from e

        self._rfdetr_to_pipeline_id = {
            rfdetr_person_id: config.COCO_PERSON_ID,
            rfdetr_laptop_id: config.COCO_LAPTOP_ID,
        }
        # `classes` (from config.DETECT_CLASSES) is expressed in pipeline/Ultralytics
        # ids; translate to RF-DETR's own ids for filtering its raw output.
        pipeline_to_rfdetr_id = {v: k for k, v in self._rfdetr_to_pipeline_id.items()}
        self.classes = {pipeline_to_rfdetr_id[c] for c in classes if c in pipeline_to_rfdetr_id}

    def detect(self, frame: np.ndarray):
        """
        Run detection on a single BGR frame (as read by OpenCV).
        RF-DETR expects RGB, so we convert; returns the same
        (boxes, confidences, class_ids) tuple as src/detector.py's Detector,
        for drop-in compatibility with tracker.py / main.py. class_ids
        returned here are already remapped to the pipeline's Ultralytics/COCO ids.
        """
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = self.model.predict(rgb, threshold=self.conf)

        if detections is None or len(detections) == 0:
            return (np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int))

        boxes = detections.xyxy
        confidences = detections.confidence
        raw_class_ids = detections.class_id.astype(int)

        keep = np.isin(raw_class_ids, list(self.classes))
        boxes, confidences, raw_class_ids = boxes[keep], confidences[keep], raw_class_ids[keep]
        class_ids = np.array([self._rfdetr_to_pipeline_id[c] for c in raw_class_ids], dtype=int)
        return boxes, confidences, class_ids


def build_detector():
    """
    Factory: returns a Detector instance based on config.DETECTOR_BACKEND,
    so main.py doesn't need to know which backend is active.
    """
    if config.DETECTOR_BACKEND == "rfdetr":
        return RFDETRDetector()
    elif config.DETECTOR_BACKEND == "yolo":
        from src.detector import Detector
        return Detector()
    else:
        raise ValueError(f"Unknown DETECTOR_BACKEND '{config.DETECTOR_BACKEND}' — use 'yolo' or 'rfdetr'")
