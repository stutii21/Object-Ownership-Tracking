"""
Zone-based ownership pipeline.

Detects ONLY the laptop each frame (no person detection/tracking at all --
that's the whole point: seats are fixed, so "owner" = nearest seat anchor
to the laptop's position, a static lookup instead of a tracking problem).

Usage:
    python main_zones.py --input data/video_slow.mp4 --output outputs/zones_annotated.mp4

If your camera/room is different, re-derive seat anchors first:
    python tools/find_seat_zones.py --input your_video.mp4
and paste the printed anchors into src/zones.py's DEFAULT_SEAT_ZONES (or
pass a custom zones list to ZoneOwnershipAssigner).
"""
from __future__ import annotations

import argparse
import time

import cv2

from config import config
from src.detector import Detector
from src.tracker import HybridTracker
from src.zones import ZoneOwnershipAssigner, DEFAULT_SEAT_ZONES
from src.handover import HandoverTracker
from src.event_logger import EventLogger


def run(input_path: str, output_path: str, show_progress_every: int = 60):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or config.FRAME_RATE
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_VIDEO_CODEC)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Only ask the detector for the laptop class -- no person detection needed.
    detector = Detector(classes=[config.COCO_LAPTOP_ID])
    tracker = HybridTracker(frame_rate=int(round(fps)))
    zone_assigner = ZoneOwnershipAssigner()
    handover_tracker = HandoverTracker()
    logger = EventLogger(csv_path="outputs/zone_events.csv", db_path="outputs/zone_events.db")

    frame_idx = 0
    start_time = time.time()
    all_events = []

    print(f"Input: {input_path} ({width}x{height} @ {fps:.2f}fps, ~{total_frames} frames)")
    print(f"Seats: {[(z.name, z.anchor_xy) for z in zone_assigner.zones]}")

    # canonical laptop id -- single-object mode, since zone assignment only
    # makes sense for one laptop at a time in this scene (matches what every
    # prior method measured: exactly one laptop identity across the clip).
    CANONICAL_LAPTOP_ID = 1000

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp_sec = frame_idx / fps

        boxes, confidences, class_ids = detector.detect(frame)
        tracked = tracker.update(frame, boxes, confidences, class_ids)

        laptop_bbox = None
        if tracked.tracker_id is not None and len(tracked) > 0:
            # take the highest-confidence laptop box this frame
            best_i = int(tracked.confidence.argmax())
            laptop_bbox = tuple(tracked.xyxy[best_i].tolist())

        annotated = frame.copy()
        # always draw the fixed seat anchors, so it's visually obvious this
        # approach doesn't depend on detecting people at all
        for zone in zone_assigner.zones:
            cx, cy = int(zone.anchor_xy[0]), int(zone.anchor_xy[1])
            cv2.circle(annotated, (cx, cy), 8, (255, 180, 0), -1)
            cv2.putText(annotated, zone.name.replace("_", " "), (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1)

        if laptop_bbox is not None:
            x1, y1, x2, y2 = laptop_bbox
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            zone_name, dist = zone_assigner.assign(centroid)

            winners = {CANONICAL_LAPTOP_ID: (zone_name, 1.0 - min(dist / zone_assigner.max_distance, 1.0))}
            events = handover_tracker.update(frame_idx, timestamp_sec, winners)
            for ev in events:
                logger.log(ev)
                all_events.append(ev)
                tag = {"init": "INIT", "handover": "HANDOVER", "unattended": "UNATTENDED"}[ev.event_type]
                print(f"[{ev.timestamp_sec:7.2f}s] frame {ev.frame_idx:5d} | {tag}: "
                      f"{ev.previous_owner} -> {ev.new_owner}")

            confirmed_zone = handover_tracker.get_owner(CANONICAL_LAPTOP_ID)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"laptop | zone: {confirmed_zone or 'unattended'} (raw nearest: {zone_name}, {dist:.0f}px)"
            cv2.putText(annotated, label, (int(x1), max(0, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            if confirmed_zone is not None:
                zone_xy = next(z.anchor_xy for z in zone_assigner.zones if z.name == confirmed_zone)
                cv2.line(annotated, (int(centroid[0]), int(centroid[1])),
                         (int(zone_xy[0]), int(zone_xy[1])), (0, 0, 255), 2)

        writer.write(annotated)
        frame_idx += 1
        if frame_idx % show_progress_every == 0:
            elapsed = time.time() - start_time
            print(f"... {frame_idx}/{total_frames} frames ({frame_idx/elapsed:.1f} fps)")

    cap.release()
    writer.release()
    logger.close()
    elapsed = time.time() - start_time
    print(f"\nDone. {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps).")
    print(f"Annotated video: {output_path}")
    print(f"Events: outputs/zone_events.csv ({len(all_events)} confirmed)")
    return all_events


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="outputs/zones_annotated.mp4")
    p.add_argument("--show-progress-every", type=int, default=60)
    args = p.parse_args()
    run(args.input, args.output, args.show_progress_every)
