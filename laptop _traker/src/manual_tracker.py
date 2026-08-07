"""
Manual bounding-box tracker.

Instead of relying on a detector (YOLO/RF-DETR) to find every person and
the laptop every frame -- which is exactly what struggles in your footage
(crowded scene, 6-10 people, laptop confidence 0.15-0.4) -- this lets YOU
draw a box once around only the object(s) you actually care about (the
laptop, and the specific people who might hold it), and then a classical
CV tracker (CSRT) follows those boxes frame-to-frame. Everyone/everything
else in the scene is never even considered, by construction.

Trade-off, stated honestly: CSRT tracks appearance+position frame-to-frame
using the box you gave it -- it does NOT re-detect on its own after a full
occlusion the way the appearance re-ID in src/reid.py does for detector-based
tracks. If a person or the laptop is fully hidden for a while and CSRT's
confidence drops, you re-draw the box once tracking resumes (this script
supports that -- see below) rather than expecting automatic recovery.

Two ways to use this module:

1. INTERACTIVE (run on your own machine, needs a display):
       python select_and_track.py --input your_video.mp4
   A window opens on the first frame. Drag a box around the laptop, press
   ENTER, type a label when prompted in the console (e.g. "laptop"). Repeat
   for each person you care about (e.g. "person_A", "person_B"). Press ESC
   when done selecting. The video then plays forward, tracking only those
   boxes and feeding them through the same ownership/handover pipeline as
   main.py. If a track visibly drifts or fails, pause (SPACE), press 'r',
   drag a fresh box over the object, and playback resumes tracking from
   there under the SAME label (so the event log stays continuous).

2. PROGRAMMATIC (no display needed -- e.g. this sandbox, or a batch job):
   Supply the initial boxes yourself as (frame_idx, label, class_id, xyxy)
   tuples -- see tests/test_manual_tracker.py for a worked example that
   bootstraps boxes from a known-good detector frame and tracks forward
   through your actual uploaded video.
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


@dataclass
class ManualTrack:
    label: str          # human-chosen name, e.g. "laptop", "person_A"
    class_id: int       # config.COCO_PERSON_ID or config.COCO_LAPTOP_ID
    tracker: any         # cv2 CSRT tracker instance
    last_bbox: tuple     # last known xyxy
    ok: bool = True      # whether the tracker still believes it's tracking correctly


class ManualMultiTracker:
    def __init__(self):
        self.tracks: dict[str, ManualTrack] = {}
        # stable integer id per label, so this plugs into the same
        # TrackedBox / OwnershipAssigner / HandoverTracker pipeline as the
        # detector-based path (which expects integer tracker_ids).
        self._label_to_id: dict[str, int] = {}
        self._next_id = 1

    def _id_for(self, label: str) -> int:
        if label not in self._label_to_id:
            self._label_to_id[label] = self._next_id
            self._next_id += 1
        return self._label_to_id[label]

    def add_box(self, frame: np.ndarray, label: str, class_id: int, xyxy: tuple):
        """Register (or re-register, after a drift/failure) a manually-drawn box."""
        x1, y1, x2, y2 = xyxy
        w, h = x2 - x1, y2 - y1
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, (x1, y1, w, h))  # CSRT wants (x, y, w, h)
        self.tracks[label] = ManualTrack(label=label, class_id=class_id, tracker=tracker,
                                          last_bbox=xyxy, ok=True)

    def update(self, frame: np.ndarray):
        """
        Advance every registered track by one frame.
        Returns a list of (stable_id, label, class_id, xyxy, ok) tuples.
        `ok=False` means CSRT lost confidence in this box THIS frame --
        the caller should keep using `last_bbox` for continuity but flag it
        for the user to re-draw (interactive mode does this automatically).
        """
        results = []
        for label, track in self.tracks.items():
            ok, (x, y, w, h) = track.tracker.update(frame)
            if ok:
                track.last_bbox = (x, y, x + w, y + h)
            track.ok = ok
            results.append((self._id_for(label), label, track.class_id, track.last_bbox, ok))
        return results
