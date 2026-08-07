"""
Re-derive seat zone anchors for a NEW video/camera (the hardcoded ones in
src/zones.py were measured specifically from video_slow.mp4).

Samples person detections across several frames spread through the video,
clusters their centroids (simple greedy clustering by distance -- good
enough for "a handful of fixed seats"), and prints ready-to-paste
SeatZone(...) lines plus a sanity check of how much each cluster's points
actually varied (a large spread means that "seat" isn't actually fixed,
and the zone-based approach is a poor fit for that camera/scene).

Usage:
    python tools/find_seat_zones.py --input your_video.mp4 --num-samples 10
"""
import argparse
import sys
import os

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from src.detector import Detector


def cluster_points(points, max_cluster_radius=60.0):
    clusters = []  # list of lists of points
    for p in points:
        placed = False
        for cluster in clusters:
            cx = np.mean([c[0] for c in cluster])
            cy = np.mean([c[1] for c in cluster])
            if np.hypot(p[0] - cx, p[1] - cy) <= max_cluster_radius:
                cluster.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
    return clusters


def run(input_path: str, num_samples: int = 10, max_cluster_radius: float = 60.0):
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frames = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    detector = Detector(classes=[config.COCO_PERSON_ID])
    all_points = []
    for fidx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ok, frame = cap.read()
        if not ok:
            continue
        boxes, confs, cls_ids = detector.detect(frame)
        for b in boxes:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            all_points.append((cx, cy))
    cap.release()

    clusters = cluster_points(all_points, max_cluster_radius)
    clusters = [c for c in clusters if len(c) >= max(2, num_samples // 3)]  # drop noise/transients
    clusters.sort(key=lambda c: np.mean([p[0] for p in c]))  # left to right

    print(f"Sampled {len(all_points)} person detections across {len(sample_frames)} frames.")
    print(f"Found {len(clusters)} stable seat clusters (appeared in >= {max(2, num_samples // 3)} samples):\n")
    print("DEFAULT_SEAT_ZONES = [")
    for i, cluster in enumerate(clusters):
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        cx, cy = np.mean(xs), np.mean(ys)
        spread = np.mean([np.hypot(x - cx, y - cy) for x, y in cluster])
        name = f"Seat_{chr(65 + i)}"
        print(f'    SeatZone("{name}", ({cx:.0f}, {cy:.0f})),'
              f"  # seen {len(cluster)}/{len(sample_frames)} samples, avg spread {spread:.0f}px")
    print("]")
    print("\nIf 'avg spread' is large (>40-50px) for a seat, that person moved around --")
    print("the zone-based approach will be less reliable for that seat specifically.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--max-cluster-radius", type=float, default=60.0)
    args = p.parse_args()
    run(args.input, args.num_samples, args.max_cluster_radius)
