"""
Ownership assignment module.

For every tracked laptop, decides which tracked person currently "owns" it
(i.e. is responsible for / actively using it), based on:

    score(person, laptop) = W_DIST * proximity_term
                           + W_IOU  * overlap_term
                           + W_TEMP * temporal_consistency_term

- proximity_term: closer person-laptop centroid distance -> higher score
                   (normalized by the frame diagonal so it's resolution independent)
- overlap_term:   IoU between an expanded "reach zone" around the person and the
                   laptop box (a person leaning over a laptop overlaps it more
                   than someone just standing nearby)
- temporal_consistency_term: a small bonus for whoever already owns the laptop,
                   so a momentary occlusion or a person passing by doesn't cause
                   ownership to flicker frame-to-frame.

This module only computes a candidate "instantaneous winner" per frame.
Actually flipping the recorded owner (and logging a handover event) is the
job of `handover.py`, which requires the winner to hold the lead for
`HANDOVER_CONFIRMATION_FRAMES` consecutive frames first.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


@dataclass
class TrackedBox:
    tracker_id: int
    xyxy: tuple  # (x1, y1, x2, y2)

    @property
    def centroid(self):
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _expand_box(box, factor: float = 1.6):
    """Expand a person's box outward to approximate their 'reach zone'."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = w * factor, h * factor
    return (cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2)


class OwnershipAssigner:
    def __init__(self, frame_diag: float):
        self.frame_diag = frame_diag
        # laptop_tracker_id -> current confirmed owner person_tracker_id (or None)
        self.current_owner: dict[int, int | None] = {}

    def score(self, person: TrackedBox, laptop: TrackedBox, laptop_id: int) -> float:
        px, py = person.centroid
        lx, ly = laptop.centroid
        dist = math.hypot(px - lx, py - ly)
        norm_dist = dist / self.frame_diag if self.frame_diag > 0 else dist
        proximity_term = 1.0 / (1.0 + norm_dist * 5.0)  # steeper falloff with distance

        overlap_term = _iou(_expand_box(person.xyxy), laptop.xyxy)

        is_current_owner = self.current_owner.get(laptop_id) == person.tracker_id
        temporal_term = 1.0 if is_current_owner else 0.0

        return (config.W_DIST * proximity_term
                + config.W_IOU * overlap_term
                + config.W_TEMP * temporal_term)

    def instantaneous_winners(self, people: list[TrackedBox], laptops: list[TrackedBox]):
        """
        Returns {laptop_tracker_id: (best_person_tracker_id_or_None, best_score)}
        for the current frame only (no temporal confirmation applied here).
        """
        results = {}
        for laptop in laptops:
            if not people:
                results[laptop.tracker_id] = (None, 0.0)
                continue
            scored = [(p.tracker_id, self.score(p, laptop, laptop.tracker_id)) for p in people]
            best_id, best_score = max(scored, key=lambda t: t[1])
            if best_score < config.MIN_OWNERSHIP_SCORE:
                results[laptop.tracker_id] = (None, best_score)
            else:
                results[laptop.tracker_id] = (best_id, best_score)
        return results
