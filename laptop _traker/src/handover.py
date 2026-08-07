"""
Handover detection module.

Consumes the per-frame "instantaneous winner" from OwnershipAssigner and
only confirms an ownership change once a candidate has been the winner for
`HANDOVER_CONFIRMATION_FRAMES` consecutive frames. This debounces
single-frame noise (occlusion, a person briefly reaching across, detector
jitter) so the event log reflects real handovers, not flicker.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


@dataclass
class HandoverEvent:
    frame_idx: int
    timestamp_sec: float
    laptop_id: int
    previous_owner: int | None
    new_owner: int | None
    confidence: float
    event_type: str  # "init" | "handover" | "unattended"


class HandoverTracker:
    def __init__(self):
        # laptop_id -> confirmed current owner (None = unattended)
        self.confirmed_owner: dict[int, int | None] = {}
        # laptop_id -> {candidate_person_id: consecutive_frame_count}
        self._candidate_streak: dict[int, dict] = {}

    def update(self, frame_idx: int, timestamp_sec: float,
               winners: dict[int, tuple]) -> list[HandoverEvent]:
        """
        winners: {laptop_id: (candidate_person_id_or_None, score)} for this frame,
                 as produced by OwnershipAssigner.instantaneous_winners().

        Returns a list of HandoverEvent objects for any change confirmed THIS frame.
        """
        events = []

        for laptop_id, (candidate, score) in winners.items():
            current = self.confirmed_owner.get(laptop_id, "__unset__")
            streaks = self._candidate_streak.setdefault(laptop_id, {})

            if current == "__unset__":
                # First time we've seen this laptop: assign immediately, no streak needed.
                self.confirmed_owner[laptop_id] = candidate
                events.append(HandoverEvent(
                    frame_idx, timestamp_sec, laptop_id,
                    previous_owner=None, new_owner=candidate,
                    confidence=score, event_type="init" if candidate is not None else "unattended",
                ))
                continue

            if candidate == current:
                # No change; reset any competing streaks.
                streaks.clear()
                continue

            # candidate differs from confirmed owner -> build up a streak before confirming
            streaks[candidate] = streaks.get(candidate, 0) + 1
            # decay other candidates' streaks
            for other in list(streaks.keys()):
                if other != candidate:
                    streaks[other] = 0

            if streaks[candidate] >= config.HANDOVER_CONFIRMATION_FRAMES:
                previous = current
                self.confirmed_owner[laptop_id] = candidate
                streaks.clear()
                events.append(HandoverEvent(
                    frame_idx, timestamp_sec, laptop_id,
                    previous_owner=previous, new_owner=candidate,
                    confidence=score,
                    event_type="handover" if candidate is not None else "unattended",
                ))

        return events

    def get_owner(self, laptop_id: int):
        return self.confirmed_owner.get(laptop_id)
