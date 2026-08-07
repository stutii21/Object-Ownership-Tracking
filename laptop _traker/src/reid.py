"""
Lightweight appearance re-identification.

ByteTrack (and most trackers) associate detections frame-to-frame using
motion + IoU overlap. That breaks down whenever an object is picked up,
carried out of its predicted position, and reappears somewhere that
doesn't overlap its last known box (exactly what happens with a laptop
being handed to someone else). When that happens, the tracker assigns a
brand-new ID, and naively you lose continuity.

Rather than blindly assuming "there's only one laptop, so merge every
laptop detection into one ID" (which breaks the moment there are two
laptops in frame), this module computes a real appearance descriptor
(HSV color histogram) for every detected box and compares descriptors
to decide whether a reappearing detection is *actually* the same
physical object, or a different one. This is the same core idea behind
DeepSORT/BoT-SORT's appearance embedding, scaled down to something that
runs on CPU with no extra model.
"""
from __future__ import annotations

import cv2
import numpy as np


def extract_histogram(frame: np.ndarray, xyxy: tuple, bins: int = 32) -> np.ndarray:
    """HSV color histogram descriptor for a bounding box crop."""
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros(bins * 2, dtype=np.float32)

    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist = np.concatenate([hist_h, hist_s]).flatten()
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 0 else hist


def similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Cosine similarity in [0, 1], 1 = identical appearance."""
    if hist_a.size == 0 or hist_b.size == 0:
        return 0.0
    denom = (np.linalg.norm(hist_a) * np.linalg.norm(hist_b))
    if denom == 0:
        return 0.0
    return float(np.clip(np.dot(hist_a, hist_b) / denom, 0.0, 1.0))


class AppearanceReID:
    """
    Keeps a small memory of recently-lost tracks (their last known
    appearance descriptor + class) and matches new, unmatched detections
    of the same class against that memory before minting a brand-new ID.
    """

    def __init__(self, similarity_threshold: float = 0.75, memory_frames: int = 90):
        self.similarity_threshold = similarity_threshold
        self.memory_frames = memory_frames
        # tracker_id -> {"hist": np.ndarray, "class_id": int, "last_seen_frame": int}
        self._memory: dict[int, dict] = {}

    def update_memory(self, tracker_id: int, class_id: int, hist: np.ndarray, frame_idx: int):
        self._memory[tracker_id] = {"hist": hist, "class_id": class_id, "last_seen_frame": frame_idx}

    def find_match(self, class_id: int, hist: np.ndarray, frame_idx: int,
                    exclude: set[int] | None = None) -> int | None:
        """
        Returns the best matching lost tracker_id if its appearance is similar
        enough and it was lost recently enough, else None. `exclude` should be
        the set of stable IDs already claimed by another detection in the
        SAME frame being processed, so we never merge two distinct objects
        that are simultaneously visible.
        """
        exclude = exclude or set()
        best_id, best_sim = None, self.similarity_threshold
        for tid, entry in self._memory.items():
            if tid in exclude:
                continue
            if entry["class_id"] != class_id:
                continue
            if frame_idx - entry["last_seen_frame"] > self.memory_frames:
                continue
            sim = similarity(hist, entry["hist"])
            if sim >= best_sim:
                best_id, best_sim = tid, sim
        return best_id

    def prune(self, frame_idx: int):
        self._memory = {
            tid: e for tid, e in self._memory.items()
            if frame_idx - e["last_seen_frame"] <= self.memory_frames
        }
