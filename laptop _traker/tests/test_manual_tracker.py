"""
Validates the manual-box tracking path end-to-end against your REAL video,
without needing a display: instead of you dragging boxes in a window, we
bootstrap the initial boxes from a frame where the detector confidently
found the laptop and its two nearest people (frame 155, t=5.16s) -- then
CSRT tracks those exact 3 boxes forward through the rest of the clip,
completely ignoring the other 5-7 people in the room, and feeds the result
through the same ownership/handover pipeline as main.py.

This tells us honestly whether "just draw it yourself and track" actually
holds up on this footage, rather than assuming it would.

Run: python tests/test_manual_tracker.py
"""
import sys, os, math, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from config import config
from src.manual_tracker import ManualMultiTracker
from src.ownership import OwnershipAssigner, TrackedBox
from src.handover import HandoverTracker

VIDEO = "/mnt/user-data/uploads/video_slow.mp4"
START_FRAME = 155  # first frame where the laptop was confidently detected (conf 0.40)

# Real boxes read off the detector at frame 155 (xyxy), used here as a stand-in
# for "the user dragged a box around these" -- same downstream code path.
INITIAL_BOXES = {
    "laptop":    (config.COCO_LAPTOP_ID, (439, 269, 534, 319)),   # conf 0.40
    "person_A":  (config.COCO_PERSON_ID, (387, 57, 547, 516)),    # nearest person, conf 0.80
    "person_B":  (config.COCO_PERSON_ID, (315, 114, 505, 623)),   # 2nd-nearest, conf 0.81
}

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_diag = math.hypot(width, height)

cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
ok, frame0 = cap.read()
assert ok

mtracker = ManualMultiTracker()
for label, (class_id, xyxy) in INITIAL_BOXES.items():
    mtracker.add_box(frame0, label, class_id, xyxy)

assigner = OwnershipAssigner(frame_diag=frame_diag)
handover_tracker = HandoverTracker()

frame_idx = START_FRAME
loss_counts = {label: 0 for label in INITIAL_BOXES}
all_events = []
t0 = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1
    timestamp_sec = frame_idx / fps

    results = mtracker.update(frame)
    people, laptops = [], []
    for stable_id, label, class_id, xyxy, track_ok in results:
        if not track_ok:
            loss_counts[label] += 1
            continue
        if class_id == config.COCO_PERSON_ID:
            people.append(TrackedBox(tracker_id=stable_id, xyxy=xyxy))
        else:
            laptops.append(TrackedBox(tracker_id=stable_id, xyxy=xyxy))

    winners = assigner.instantaneous_winners(people, laptops)
    events = handover_tracker.update(frame_idx, timestamp_sec, winners)
    all_events.extend(events)
    for laptop in laptops:
        assigner.current_owner[laptop.tracker_id] = handover_tracker.get_owner(laptop.tracker_id)

cap.release()
elapsed = time.time() - t0
n_processed = frame_idx - START_FRAME

print(f"Processed frames {START_FRAME}->{frame_idx} ({n_processed} frames) in {elapsed:.1f}s "
      f"({n_processed/elapsed:.1f} fps -- CSRT-only, no detector inference cost)")
print(f"\nCSRT tracking-loss counts (frames where CSRT itself reported low confidence):")
for label, count in loss_counts.items():
    pct = 100 * count / n_processed if n_processed else 0
    print(f"  {label}: lost on {count}/{n_processed} frames ({pct:.1f}%)")

print(f"\nConfirmed events ({len(all_events)}):")
for ev in all_events:
    print(f"  [{ev.timestamp_sec:6.2f}s] frame {ev.frame_idx} laptop={ev.laptop_id} "
          f"{ev.previous_owner}->{ev.new_owner} type={ev.event_type} conf={ev.confidence:.2f}")
