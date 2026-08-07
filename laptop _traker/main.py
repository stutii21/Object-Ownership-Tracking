"""
Main entry point.

Usage:
    python main.py --input path/to/your_video.mp4 --output outputs/annotated.mp4

Reads an input video, runs:
    detection (YOLO, pretrained COCO: person + laptop)
      -> tracking (ByteTrack, persistent IDs)
      -> ownership assignment (proximity + overlap + temporal consistency)
      -> handover detection (debounced ownership-change events)
and produces:
    - an annotated .mp4 with boxes, tracker IDs, and an "Owner: <id>" label per laptop
    - outputs/events.csv   (every confirmed handover / init / unattended event)
    - outputs/events.db    (same, in SQLite, for the dashboard)
"""
from __future__ import annotations

import argparse
import math
import time

import cv2
import numpy as np
import supervision as sv

from config import config
from src.detector_rfdetr import build_detector
from src.tracker import HybridTracker
from src.ownership import OwnershipAssigner, TrackedBox
from src.handover import HandoverTracker
from src.event_logger import EventLogger


def parse_args():
    p = argparse.ArgumentParser(description="Laptop ownership & handover tracking pipeline")
    p.add_argument("--input", required=True, help="Path to input .mp4 file")
    p.add_argument("--output", default=config.ANNOTATED_VIDEO_PATH, help="Path to write annotated .mp4")
    p.add_argument("--show-progress-every", type=int, default=60,
                   help="Print progress every N frames")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Optional cap on frames processed (useful for quick tests)")
    return p.parse_args()


def run(input_path: str, output_path: str, show_progress_every: int = 60, max_frames: int | None = None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or config.FRAME_RATE
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_diag = math.hypot(width, height)

    fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_VIDEO_CODEC)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector = build_detector()
    print(f"Detector backend: {config.DETECTOR_BACKEND}"
          + (f" ({config.RFDETR_VARIANT})" if config.DETECTOR_BACKEND == "rfdetr" else f" ({config.MODEL_WEIGHTS})"))
    tracker = HybridTracker(frame_rate=int(round(fps)))
    assigner = OwnershipAssigner(frame_diag=frame_diag)
    handover_tracker = HandoverTracker()
    logger = EventLogger()

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    frame_idx = 0
    start_time = time.time()
    all_events = []

    print(f"Input: {input_path}  ({width}x{height} @ {fps:.2f}fps, ~{total_frames} frames)")
    print(f"Writing annotated video to: {output_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        timestamp_sec = frame_idx / fps

        boxes, confidences, class_ids = detector.detect(frame)
        tracked: sv.Detections = tracker.update(frame, boxes, confidences, class_ids)

        people, laptops = [], []
        if tracked.tracker_id is not None:
            for i in range(len(tracked)):
                tid = int(tracked.tracker_id[i])
                cls = int(tracked.class_id[i])
                xyxy = tuple(tracked.xyxy[i].tolist())
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                if cls == config.COCO_PERSON_ID:
                    people.append(TrackedBox(tracker_id=tid, xyxy=xyxy))
                elif cls == config.COCO_LAPTOP_ID:
                    laptops.append((conf, TrackedBox(tracker_id=tid, xyxy=xyxy)))

        # Track continuity across occlusion/pickup gaps is now handled by
        # HybridTracker's appearance re-ID (src/reid.py), not by blindly
        # assuming one object. SINGLE_OBJECT_MODE here only resolves a
        # DIFFERENT, legitimate ambiguity: if the detector fires two "laptop"
        # boxes in the very same frame (duplicate/false-positive box) while
        # you know there's physically only one, keep the higher-confidence box.
        if config.SINGLE_OBJECT_MODE and len(laptops) > 1:
            laptops = [max(laptops, key=lambda t: t[0])]
        laptops = [tb for _, tb in laptops]

        winners = assigner.instantaneous_winners(people, laptops)
        events = handover_tracker.update(frame_idx, timestamp_sec, winners)
        for ev in events:
            logger.log(ev)
            all_events.append(ev)
            tag = {"init": "INIT", "handover": "HANDOVER", "unattended": "UNATTENDED"}[ev.event_type]
            print(f"[{ev.timestamp_sec:7.2f}s] frame {ev.frame_idx:5d} | laptop {ev.laptop_id} "
                  f"| {tag}: {ev.previous_owner} -> {ev.new_owner} (conf={ev.confidence:.2f})")

        # keep the assigner's notion of "current owner" in sync with confirmed owner,
        # so its temporal-consistency term rewards stability rather than the noisy
        # instantaneous winner.
        for laptop in laptops:
            assigner.current_owner[laptop.tracker_id] = handover_tracker.get_owner(laptop.tracker_id)

        annotated = frame.copy()
        if len(tracked) > 0:
            labels = []
            for i in range(len(tracked)):
                tid = int(tracked.tracker_id[i])
                cls = int(tracked.class_id[i])
                if cls == config.COCO_PERSON_ID:
                    labels.append(f"Person #{tid}")
                else:
                    owner = handover_tracker.get_owner(tid)
                    labels.append(f"Laptop #{tid} | Owner: {owner if owner is not None else 'unattended'}")
            annotated = box_annotator.annotate(scene=annotated, detections=tracked)
            annotated = label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)

        # draw a connecting line from each laptop to its confirmed owner, for clarity
        people_by_id = {p.tracker_id: p for p in people}
        for laptop in laptops:
            owner_id = handover_tracker.get_owner(laptop.tracker_id)
            if owner_id is not None and owner_id in people_by_id:
                lx, ly = laptop.centroid
                px, py = people_by_id[owner_id].centroid
                cv2.line(annotated, (int(lx), int(ly)), (int(px), int(py)), (0, 0, 255), 2)

        writer.write(annotated)

        frame_idx += 1
        if frame_idx % show_progress_every == 0:
            elapsed = time.time() - start_time
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            print(f"... processed {frame_idx}/{total_frames} frames ({fps_proc:.1f} fps)")

    cap.release()
    writer.release()
    logger.close()

    elapsed = time.time() - start_time
    print(f"\nDone. Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps).")
    print(f"Annotated video: {output_path}")
    print(f"Event log (CSV): {config.EVENT_LOG_CSV}")
    print(f"Event log (SQLite): {config.EVENT_LOG_SQLITE}")
    print(f"Total confirmed events: {len(all_events)}")
    return all_events


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output, args.show_progress_every, args.max_frames)
