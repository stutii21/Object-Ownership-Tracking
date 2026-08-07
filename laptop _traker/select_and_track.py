"""
Interactive manual box selection + tracking.

Run this on YOUR OWN machine (needs a display -- won't work in a headless
server/sandbox):

    python select_and_track.py --input your_video.mp4

Controls:
    - On the first frame, a selection window opens.
    - Drag a box around the laptop. Press ENTER/SPACE to confirm.
      In the console, type a label, e.g.: laptop
    - Drag a box around each person you want to track (only the ones who
      might plausibly hold the laptop -- ignore everyone else in the room).
      Label them e.g.: person_A, person_B, ...
    - Press ESC in the selection window when you're done adding boxes.
    - Playback then runs, tracking only the boxes you drew, and logging
      ownership/handover exactly like main.py does.
    - If a box visibly drifts off its object (CSRT lost it), press SPACE to
      pause, press 'r', drag a fresh box over the object in the paused
      frame, press ENTER, and playback resumes -- same label, so the event
      log stays continuous rather than starting a "new" object.
    - Press 'q' to stop early and finalize the output/log with whatever was
      tracked so far.

NO-DISPLAY / HEADLESS MODE:
If you don't have a display available (e.g. running on a server, or
you already know the box coordinates from another tool like makesense.ai),
skip interactive selection entirely with --boxes-file:

    python select_and_track.py --input your_video.mp4 --boxes-file boxes.json --start-frame 155

boxes.json format:
    {
      "start_frame": 155,
      "boxes": {
        "laptop":   {"class": "laptop", "xyxy": [439, 269, 534, 319]},
        "person_A": {"class": "person", "xyxy": [387, 57, 547, 516]},
        "person_B": {"class": "person", "xyxy": [315, 114, 505, 623]}
      }
    }
(xyxy = [x1, y1, x2, y2] in pixel coordinates on the frame at start_frame)

Output (same as main.py): outputs/annotated.mp4, outputs/events.csv, outputs/events.db
"""
from __future__ import annotations

import argparse
import json
import math
import time

import cv2
import supervision as sv

from config import config
from src.manual_tracker import ManualMultiTracker
from src.ownership import OwnershipAssigner, TrackedBox
from src.handover import HandoverTracker
from src.event_logger import EventLogger

CLASS_NAME_TO_ID = {"laptop": config.COCO_LAPTOP_ID, "person": config.COCO_PERSON_ID}


def prompt_label_and_class(box_num: int):
    print(f"\nBox #{box_num} selected.")
    label = input("  Label for this box (e.g. 'laptop', 'person_A'): ").strip()
    if label.lower().startswith("laptop"):
        class_id = config.COCO_LAPTOP_ID
    else:
        class_id = config.COCO_PERSON_ID
    return label, class_id


def select_initial_boxes(frame):
    boxes = {}
    print("\n=== Draw a box around each object you want tracked ===")
    print("Drag a box, press ENTER/SPACE to confirm each one. Press ESC when done.\n")
    n = 0
    while True:
        x, y, w, h = cv2.selectROI("Select object (ESC when done)", frame, showCrosshair=True)
        if w == 0 or h == 0:
            break
        n += 1
        label, class_id = prompt_label_and_class(n)
        boxes[label] = (class_id, (x, y, x + w, y + h))
    cv2.destroyWindow("Select object (ESC when done)")
    return boxes


def run(input_path: str, output_path: str = config.ANNOTATED_VIDEO_PATH,
        boxes_file: str = None, start_frame: int = 0):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or config.FRAME_RATE
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_diag = math.hypot(width, height)

    if boxes_file:
        with open(boxes_file) as f:
            spec = json.load(f)
        start_frame = spec.get("start_frame", start_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, first_frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {start_frame}.")
        initial_boxes = {
            label: (CLASS_NAME_TO_ID.get(box["class"], config.COCO_PERSON_ID), tuple(box["xyxy"]))
            for label, box in spec["boxes"].items()
        }
        interactive = False
    else:
        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, first_frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {start_frame}.")
        initial_boxes = select_initial_boxes(first_frame)
        interactive = True

    if not initial_boxes:
        print("No boxes selected -- nothing to track. Exiting.")
        return

    mtracker = ManualMultiTracker()
    for label, (class_id, xyxy) in initial_boxes.items():
        mtracker.add_box(first_frame, label, class_id, xyxy)

    fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_VIDEO_CODEC)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    assigner = OwnershipAssigner(frame_diag=frame_diag)
    handover_tracker = HandoverTracker()
    logger = EventLogger()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    frame_idx = start_frame

    def process_frame(frame, frame_idx):
        timestamp_sec = frame_idx / fps
        results = mtracker.update(frame)

        people, laptops = [], []
        for stable_id, label, class_id, xyxy, ok in results:
            if not ok:
                continue  # skip feeding a lost box into ownership scoring this frame
            if class_id == config.COCO_PERSON_ID:
                people.append(TrackedBox(tracker_id=stable_id, xyxy=xyxy))
            else:
                laptops.append(TrackedBox(tracker_id=stable_id, xyxy=xyxy))

        winners = assigner.instantaneous_winners(people, laptops)
        events = handover_tracker.update(frame_idx, timestamp_sec, winners)
        for ev in events:
            logger.log(ev)
            tag = {"init": "INIT", "handover": "HANDOVER", "unattended": "UNATTENDED"}[ev.event_type]
            print(f"[{ev.timestamp_sec:7.2f}s] laptop {ev.laptop_id} | {tag}: "
                  f"{ev.previous_owner} -> {ev.new_owner}")

        for laptop in laptops:
            assigner.current_owner[laptop.tracker_id] = handover_tracker.get_owner(laptop.tracker_id)

        annotated = frame.copy()
        for stable_id, label, class_id, xyxy, ok in results:
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            color = (0, 200, 0) if ok else (0, 0, 255)
            tag = label if ok else f"{label} (LOST - press SPACE, 'r' to redraw)"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, tag, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return annotated

    writer.write(process_frame(first_frame, frame_idx))
    frame_idx += 1

    if not interactive:
        # Headless mode: stream straight through, no window/keyboard involved.
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(process_frame(frame, frame_idx))
            frame_idx += 1
    else:
        paused = False
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated = process_frame(frame, frame_idx)
                writer.write(annotated)
                cv2.imshow("Tracking (SPACE=pause, r=redraw a box, q=quit)", annotated)
                frame_idx += 1
            key = cv2.waitKey(1 if not paused else 50) & 0xFF
            if key == ord(' '):
                paused = not paused
            elif key == ord('q'):
                break
            elif key == ord('r') and paused:
                x, y, w, h = cv2.selectROI("Redraw box", frame, showCrosshair=True)
                cv2.destroyWindow("Redraw box")
                if w > 0 and h > 0:
                    label = input("  Label to re-attach this box to (existing label to keep continuity): ").strip()
                    class_id = CLASS_NAME_TO_ID.get(label.split('_')[0], config.COCO_PERSON_ID)
                    mtracker.add_box(frame, label, class_id, (x, y, x + w, y + h))

    cap.release()
    writer.release()
    if interactive:
        cv2.destroyAllWindows()
    logger.close()
    print(f"\nDone. Annotated video: {output_path}")
    print(f"Event log: {config.EVENT_LOG_CSV} / {config.EVENT_LOG_SQLITE}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=config.ANNOTATED_VIDEO_PATH)
    p.add_argument("--boxes-file", default=None,
                   help="Path to a boxes.json (see module docstring) to skip interactive "
                        "selection -- use this on a machine/sandbox with no display.")
    p.add_argument("--start-frame", type=int, default=0,
                   help="Frame index to start on (ignored if --boxes-file sets its own start_frame)")
    args = p.parse_args()
    run(args.input, args.output, boxes_file=args.boxes_file, start_frame=args.start_frame)
