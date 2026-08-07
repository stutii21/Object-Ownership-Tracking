"""
Full measurement pass over the real input video. Captures everything worth
reporting: per-class detection counts/confidence stats, unique track counts,
laptop visibility rate, ownership scoring trace, confirmed events, and
timing/throughput -- then writes a single JSON report.

Run: python measure.py --input /path/to/video.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import time
import statistics

import cv2
import numpy as np

from config import config
from src.detector_rfdetr import build_detector
from src.tracker import HybridTracker
from src.ownership import OwnershipAssigner, TrackedBox
from src.handover import HandoverTracker


def run(input_path: str, report_path: str = "outputs/measurement_report.json"):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or config.FRAME_RATE
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_diag = math.hypot(width, height)

    detector = build_detector()
    tracker = HybridTracker(frame_rate=int(round(fps)))
    assigner = OwnershipAssigner(frame_diag=frame_diag)
    handover_tracker = HandoverTracker()

    person_confidences, laptop_confidences = [], []
    person_counts_per_frame, laptop_counts_per_frame = [], []
    frames_with_laptop = 0
    all_person_track_ids = set()
    all_laptop_track_ids_raw = set()
    per_frame_top_score = []  # (frame_idx, timestamp, winner_person_id, score) whenever a laptop is seen
    all_events = []

    frame_idx = 0
    t0 = time.time()
    per_frame_infer_times = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp_sec = frame_idx / fps

        t_start = time.time()
        boxes, confidences, class_ids = detector.detect(frame)
        per_frame_infer_times.append(time.time() - t_start)

        tracked = tracker.update(frame, boxes, confidences, class_ids)

        people, laptops_scored = [], []
        n_person_this_frame, n_laptop_this_frame = 0, 0
        if tracked.tracker_id is not None:
            for i in range(len(tracked)):
                tid = int(tracked.tracker_id[i])
                cls = int(tracked.class_id[i])
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else None
                xyxy = tuple(tracked.xyxy[i].tolist())
                if cls == config.COCO_PERSON_ID:
                    people.append(TrackedBox(tracker_id=tid, xyxy=xyxy))
                    all_person_track_ids.add(tid)
                    n_person_this_frame += 1
                    if conf is not None:
                        person_confidences.append(conf)
                elif cls == config.COCO_LAPTOP_ID:
                    all_laptop_track_ids_raw.add(tid)
                    n_laptop_this_frame += 1
                    if conf is not None:
                        laptop_confidences.append(conf)
                    laptops_scored.append((conf or 0.0, TrackedBox(tracker_id=tid, xyxy=xyxy)))

        person_counts_per_frame.append(n_person_this_frame)
        laptop_counts_per_frame.append(n_laptop_this_frame)
        if n_laptop_this_frame > 0:
            frames_with_laptop += 1

        # Track continuity across gaps is handled by HybridTracker's appearance
        # re-ID now. SINGLE_OBJECT_MODE here only dedupes true same-frame
        # duplicate detections (keep the higher-confidence box).
        if config.SINGLE_OBJECT_MODE and len(laptops_scored) > 1:
            laptops_scored = [max(laptops_scored, key=lambda t: t[0])]
        laptops = [tb for _, tb in laptops_scored]

        winners = assigner.instantaneous_winners(people, laptops)
        for laptop_id, (winner, score) in winners.items():
            per_frame_top_score.append((frame_idx, round(timestamp_sec, 2), winner, round(score, 3)))

        events = handover_tracker.update(frame_idx, timestamp_sec, winners)
        for ev in events:
            all_events.append({
                "frame": ev.frame_idx, "t_sec": round(ev.timestamp_sec, 2),
                "laptop_id": ev.laptop_id, "previous_owner": ev.previous_owner,
                "new_owner": ev.new_owner, "confidence": round(ev.confidence, 3),
                "type": ev.event_type,
            })

        for laptop in laptops:
            assigner.current_owner[laptop.tracker_id] = handover_tracker.get_owner(laptop.tracker_id)

        frame_idx += 1

    cap.release()
    total_time = time.time() - t0

    def stats(vals):
        if not vals:
            return {"count": 0}
        return {
            "count": len(vals),
            "mean": round(statistics.mean(vals), 3),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
            "median": round(statistics.median(vals), 3),
        }

    report = {
        "input_video": input_path,
        "video_properties": {
            "width": width, "height": height, "fps": round(fps, 2),
            "total_frames": total_frames, "duration_sec": round(total_frames / fps, 2),
        },
        "detector_backend": config.DETECTOR_BACKEND,
        "detector_model": config.RFDETR_VARIANT if config.DETECTOR_BACKEND == "rfdetr" else config.MODEL_WEIGHTS,
        "confidence_threshold": config.CONF_THRESHOLD,
        "performance": {
            "frames_processed": frame_idx,
            "total_wall_time_sec": round(total_time, 1),
            "avg_inference_time_ms": round(statistics.mean(per_frame_infer_times) * 1000, 1),
            "throughput_fps": round(frame_idx / total_time, 2),
        },
        "person_detection": {
            "unique_track_ids_seen": len(all_person_track_ids),
            "avg_people_per_frame": round(statistics.mean(person_counts_per_frame), 2),
            "max_people_in_single_frame": max(person_counts_per_frame),
            "confidence_stats": stats(person_confidences),
        },
        "laptop_detection": {
            "unique_raw_track_ids_seen": len(all_laptop_track_ids_raw),
            "note": "raw ids include track resets from occlusion/low-conf gaps; "
                    "SINGLE_OBJECT_MODE collapses these to one canonical object downstream",
            "frames_with_laptop_detected": frames_with_laptop,
            "pct_frames_with_laptop": round(100 * frames_with_laptop / frame_idx, 1),
            "confidence_stats": stats(laptop_confidences),
        },
        "ownership_assignment": {
            "confirmed_events": all_events,
            "num_confirmed_events": len(all_events),
        },
        "config_used": {
            "W_DIST": config.W_DIST, "W_IOU": config.W_IOU, "W_TEMP": config.W_TEMP,
            "MIN_OWNERSHIP_SCORE": config.MIN_OWNERSHIP_SCORE,
            "HANDOVER_CONFIRMATION_FRAMES": config.HANDOVER_CONFIRMATION_FRAMES,
            "SINGLE_OBJECT_MODE": config.SINGLE_OBJECT_MODE,
            "LOST_TRACK_BUFFER": config.LOST_TRACK_BUFFER,
        },
    }

    import os
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nFull report written to {report_path}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--report", default="outputs/measurement_report.json")
    args = p.parse_args()
    run(args.input, args.report)
