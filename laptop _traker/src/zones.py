"""
Zone-based ownership assignment.

Instead of detecting AND tracking every person every frame (which is where
most of the error in this project has come from -- low/intermittent
confidence, occlusion, ID switches), this assumes what we actually verified
on your video: seats are fixed for the whole clip. Person cluster centroids
measured at frames 0, 100, 200, 300, 400, 500, 594 stayed within ~20-30px of
each other the entire time.

Given that, "who owns the laptop" reduces to "which fixed seat is the
laptop closest to" -- a static geometry lookup, not a tracking problem.
This eliminates person-detection confidence/occlusion/ID-switch error
from the pipeline ENTIRELY for the ownership decision (person detection is
no longer even in the loop) -- only the laptop needs to be found each frame.

Trade-off, stated honestly: this assumes people don't get up and change
seats mid-video. If they do, this approach silently mis-attributes
ownership to whoever's seat the laptop is near, not to the actual moving
person. Use the detector+tracker pipeline (main.py) instead if seat-swapping
is a real possibility in your setting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SeatZone:
    name: str
    anchor_xy: tuple  # (x, y) fixed pixel position of this seat, e.g. person centroid

    def distance_to(self, point_xy: tuple) -> float:
        return math.hypot(point_xy[0] - self.anchor_xy[0], point_xy[1] - self.anchor_xy[1])


# Seat anchors measured directly from your video via tools/find_seat_zones.py
# (12-frame sample, person-detection clustering). Re-derive for a different
# video/camera by running that tool again.
DEFAULT_SEAT_ZONES = [
    SeatZone("Seat_A", (311, 534)),   # avg spread 5px across samples
    SeatZone("Seat_B", (394, 515)),   # avg spread 23px
    SeatZone("Seat_C", (417, 365)),   # avg spread 14px
    SeatZone("Seat_D", (474, 284)),   # avg spread 4px
    SeatZone("Seat_E", (525, 194)),   # avg spread 6px
    SeatZone("Seat_F", (643, 390)),   # avg spread 19px
    SeatZone("Seat_G", (653, 215)),   # avg spread 11px
]


class ZoneOwnershipAssigner:
    def __init__(self, zones: list[SeatZone] = None, max_distance: float = 220.0):
        """
        max_distance: if the laptop's centroid is farther than this (pixels)
        from every seat anchor, ownership is "unattended" instead of forcing
        an assignment to the nearest-but-still-far seat.
        """
        self.zones = zones if zones is not None else DEFAULT_SEAT_ZONES
        self.max_distance = max_distance

    def assign(self, laptop_centroid: tuple) -> tuple:
        """Returns (zone_name_or_None, distance)."""
        distances = [(z.name, z.distance_to(laptop_centroid)) for z in self.zones]
        name, dist = min(distances, key=lambda t: t[1])
        if dist > self.max_distance:
            return None, dist
        return name, dist
