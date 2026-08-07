"""
Non-interactive alternative to select_and_track.py.

If dragging boxes in a popup window is inconvenient (no display, remote
server, or you just don't like GUIs), specify the boxes in a JSON file
instead -- get the coordinates from any still-frame image viewer (even
just opening a saved frame in Preview/Paint and hovering to read pixel
coordinates), or from a free browser tool like https://www.makesense.ai.

boxes.json format:
{
  "laptop":   {"frame": 155, "class": "laptop", "xyxy": [439, 269, 534, 319]},
  "person_A": {"frame": 0,   "class": "person", "xyxy": [394, 52, 541, 513]},
  "person_B": {"frame": 0,   "class": "person", "xyxy": [317, 142, 509, 628]}
}

- "frame": which frame number this box should start being tracked from
  (use 0 if the object is visible from the start; use a later frame number
  if e.g. the laptop is out of frame or hidden until then).
- "class": "laptop" or "person".
- "xyxy": [x1, y1, x2, y2] pixel coordinates (top-left corner, then
  bottom-right corner) of the box in that frame.

Run:
    python track_from_config.py --input your_video.mp4 --boxes boxes.json
"""
from __future__ import annotations

import argparse
import json
import math

import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import config
from src.manual_tracker import ManualMultiTracker
from src.ownership import OwnershipAssigner, TrackedBox
from src.handover import HandoverTracker
from src.event_logger import EventLogger

CLASS_NAME_TO_ID = {"laptop": config.COCO_LAPTOP_ID, "person": config.COCO_PERSON_ID}


def run(input_path: str, boxes_path: str, output_path: str = config.ANNOTATED_VIDEO_PATH):
    with open(boxes_path) as f:
        box_specs = json.load(f)

    # group by the frame they should be added on
    by_add_frame: dict[int, list] = {}
    for label, spec in box_specs.items():
        by_add_frame.setdefault(spec["frame"], []).append(
            (label, CLASS_NAME_TO_ID[spec["class"]], tuple(spec["xyxy"]))
        )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or config.FRAME_RATE
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_diag = math.hypot(width, height)

    fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_VIDEO_CODEC)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    mtracker = ManualMultiTracker()
    assigner = OwnershipAssigner(frame_diag=frame_diag)
    handover_tracker = HandoverTracker()
    logger = EventLogger()

    frame_idx = 0
    pending_frames = sorted(by_add_frame.keys())

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        while pending_frames and pending_frames[0] <= frame_idx:
            fidx = pending_frames.pop(0)
            for label, class_id, xyxy in by_add_frame[fidx]:
                mtracker.add_box(frame, label, class_id, xyxy)
                print(f"[frame {frame_idx}] added box '{label}' ({['person','laptop'][class_id==config.COCO_LAPTOP_ID]})")

        timestamp_sec = frame_idx / fps
        results = mtracker.update(frame)
        people, laptops = [], []
        for stable_id, label, class_id, xyxy, ok_track in results:
            if not ok_track:
                continue
            (people if class_id == config.COCO_PERSON_ID else laptops).append(
                TrackedBox(tracker_id=stable_id, xyxy=xyxy))

        winners = assigner.instantaneous_winners(people, laptops)
        events = handover_tracker.update(frame_idx, timestamp_sec, winners)
        for ev in events:
            logger.log(ev)
            tag = {"init": "INIT", "handover": "HANDOVER", "unattended": "UNATTENDED"}[ev.event_type]
            print(f"[{ev.timestamp_sec:7.2f}s] laptop {ev.laptop_id} | {tag}: {ev.previous_owner} -> {ev.new_owner}")
        for laptop in laptops:
            assigner.current_owner[laptop.tracker_id] = handover_tracker.get_owner(laptop.tracker_id)

        annotated = frame.copy()
        for stable_id, label, class_id, xyxy, ok_track in results:
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            color = (0, 200, 0) if ok_track else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    logger.close()
    print(f"\nDone. {frame_idx} frames. Annotated video: {output_path}")
    print(f"Event log: {config.EVENT_LOG_CSV} / {config.EVENT_LOG_SQLITE}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--boxes", required=True, help="Path to boxes.json")
    p.add_argument("--output", default=config.ANNOTATED_VIDEO_PATH)
    args = p.parse_args()
    run(args.input, args.boxes, args.output)
