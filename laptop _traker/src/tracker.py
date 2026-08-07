"""
Hybrid tracker: ByteTrack (motion/IoU-based, frame-to-frame) + an
appearance re-identification fallback (src/reid.py) that specifically
handles the case ByteTrack cannot: an object is picked up, moved outside
its predicted position, and reappears. ByteTrack alone would assign a new
ID there; this module checks the new detection's appearance against
recently-lost tracks of the same class and re-attaches the OLD id if the
appearance genuinely matches, rather than assuming it must be the same
object just because "there's only one" of that class.

This generalizes correctly even if you have 2+ laptops in frame: it will
only merge IDs when appearance similarity crosses the threshold, not
just by class.
"""
from __future__ import annotations

import numpy as np
import supervision as sv

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from src.reid import AppearanceReID, extract_histogram


class HybridTracker:
    def __init__(self, frame_rate: int = config.FRAME_RATE):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.TRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer=config.LOST_TRACK_BUFFER,
            minimum_matching_threshold=config.MINIMUM_MATCHING_THRESHOLD,
            frame_rate=frame_rate,
        )
        self.reid = AppearanceReID(
            similarity_threshold=config.REID_SIMILARITY_THRESHOLD,
            memory_frames=config.REID_MEMORY_FRAMES,
        )
        # Maps ByteTrack's raw (possibly-new) IDs -> the stable ID we expose downstream.
        self._id_remap: dict[int, int] = {}
        self._frame_idx = 0

    def update(self, frame: np.ndarray, boxes: np.ndarray, confidences: np.ndarray,
               class_ids: np.ndarray) -> sv.Detections:
        detections = sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids)
        tracked = self.tracker.update_with_detections(detections)

        if tracked.tracker_id is None or len(tracked) == 0:
            self.reid.prune(self._frame_idx)
            self._frame_idx += 1
            return tracked

        stable_ids = np.zeros(len(tracked), dtype=int)
        hists = [None] * len(tracked)
        used_this_frame: set[int] = set()

        # Pass 1: raw IDs ByteTrack itself kept continuous -- these are the most
        # reliable signal and are claimed first, so re-ID (pass 2) can never
        # steal a stable ID that's genuinely still visible this frame.
        for i in range(len(tracked)):
            raw_id = int(tracked.tracker_id[i])
            xyxy = tuple(tracked.xyxy[i].tolist())
            hists[i] = extract_histogram(frame, xyxy)
            if raw_id in self._id_remap:
                stable_id = self._id_remap[raw_id]
                stable_ids[i] = stable_id
                used_this_frame.add(stable_id)

        # Pass 2: brand-new raw IDs. Check whether each looks like a recently-lost
        # object reappearing elsewhere (e.g. laptop picked up and moved to a new
        # desk), but never match onto a stable ID already claimed THIS frame --
        # that would incorrectly merge two distinct objects visible at once.
        for i in range(len(tracked)):
            raw_id = int(tracked.tracker_id[i])
            if raw_id in self._id_remap:
                continue
            cls_id = int(tracked.class_id[i])
            match = self.reid.find_match(cls_id, hists[i], self._frame_idx, exclude=used_this_frame)
            stable_id = match if match is not None else raw_id
            self._id_remap[raw_id] = stable_id
            stable_ids[i] = stable_id
            used_this_frame.add(stable_id)

        for i in range(len(tracked)):
            cls_id = int(tracked.class_id[i])
            self.reid.update_memory(int(stable_ids[i]), cls_id, hists[i], self._frame_idx)

        self.reid.prune(self._frame_idx)
        self._frame_idx += 1

        tracked.tracker_id = stable_ids
        return tracked

    def reset(self):
        self.tracker.reset()
        self._id_remap.clear()
        self._frame_idx = 0
