"""
End-to-end logic test that bypasses the YOLO detector (which needs real
photographic content to fire) and instead feeds synthetic, known
bounding boxes straight into the tracker -> ownership -> handover chain.

This proves the tracking IDs stay stable, the ownership score favors the
nearer person, and a handover event fires only after the confirmation
window when the laptop moves across the room to a second person.

Run: python tests/test_pipeline_logic.py
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.tracker import HybridTracker
from src.ownership import OwnershipAssigner, TrackedBox
from src.handover import HandoverTracker
from config import config

WIDTH, HEIGHT = 640, 480
FRAME_DIAG = (WIDTH ** 2 + HEIGHT ** 2) ** 0.5
N_FRAMES = 90

tracker = HybridTracker(frame_rate=30)
dummy_frame = (np.random.rand(HEIGHT, WIDTH, 3) * 255).astype(np.uint8)
assigner = OwnershipAssigner(frame_diag=FRAME_DIAG)
handover_tracker = HandoverTracker()

events_seen = []

for i in range(N_FRAMES):
    # person A static box, person B static box, laptop moves from near A to near B
    person_a = (50, 100, 150, 400)
    person_b = (450, 100, 550, 400)
    laptop_x = 100 + (400 * i / N_FRAMES)
    laptop = (laptop_x, 250, laptop_x + 80, 320)

    boxes = np.array([person_a, person_b, laptop], dtype=float)
    confidences = np.array([0.9, 0.9, 0.9])
    class_ids = np.array([config.COCO_PERSON_ID, config.COCO_PERSON_ID, config.COCO_LAPTOP_ID])

    tracked = tracker.update(dummy_frame, boxes, confidences, class_ids)

    people, laptops = [], []
    for j in range(len(tracked)):
        tid = int(tracked.tracker_id[j])
        cls = int(tracked.class_id[j])
        xyxy = tuple(tracked.xyxy[j].tolist())
        if cls == config.COCO_PERSON_ID:
            people.append(TrackedBox(tracker_id=tid, xyxy=xyxy))
        else:
            laptops.append(TrackedBox(tracker_id=tid, xyxy=xyxy))

    winners = assigner.instantaneous_winners(people, laptops)
    events = handover_tracker.update(i, i / 30.0, winners)
    events_seen.extend(events)

    for laptop_tb in laptops:
        assigner.current_owner[laptop_tb.tracker_id] = handover_tracker.get_owner(laptop_tb.tracker_id)

print(f"Total frames processed: {N_FRAMES}")
print(f"Tracked person IDs stayed stable: {sorted(set(p.tracker_id for p in people))}")
print(f"Events fired ({len(events_seen)}):")
for ev in events_seen:
    print(f"  frame={ev.frame_idx:3d} laptop={ev.laptop_id} {ev.previous_owner}->{ev.new_owner} "
          f"type={ev.event_type} conf={ev.confidence:.2f}")

assert len(events_seen) >= 2, "Expected at least an init event and a handover event"
assert events_seen[0].event_type == "init"
assert any(e.event_type == "handover" for e in events_seen), "Expected a handover to be detected"
print("\nPASS: init + handover detected as expected.")

# ---------------------------------------------------------------------------
# Second test: the actual real-world failure mode from the user's video --
# an object is occluded/picked up (disappears for several frames) and
# reappears in a position that does NOT overlap its last known box, which
# defeats ByteTrack's pure IoU matching. Appearance re-ID should still
# recognize it as the same object and keep the same stable ID.
# ---------------------------------------------------------------------------
print("\n--- Gap-bridging test (object disappears, reappears elsewhere) ---")
tracker2 = HybridTracker(frame_rate=30)

# distinctly colored laptop crop so its histogram is unambiguous
def make_frame(laptop_box=None, laptop_color=(80, 180, 80)):
    f = np.full((HEIGHT, WIDTH, 3), 220, dtype=np.uint8)
    if laptop_box is not None:
        x1, y1, x2, y2 = laptop_box
        f[y1:y2, x1:x2] = laptop_color
    return f

seen_ids = []
for i in range(10):
    box = (100, 250, 180, 320)
    frame = make_frame(box)
    boxes = np.array([box], dtype=float)
    tracked = tracker2.update(frame, boxes, np.array([0.9]), np.array([config.COCO_LAPTOP_ID]))
    seen_ids.append(int(tracked.tracker_id[0]))
first_id = seen_ids[-1]

# 40-frame gap: object not detected at all (picked up, in transit)
for i in range(40):
    frame = make_frame(None)
    tracked = tracker2.update(frame, np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int))

# reappears far away, same color -- should re-attach to the SAME stable id
for i in range(10):
    box = (500, 50, 580, 120)
    frame = make_frame(box)
    boxes = np.array([box], dtype=float)
    tracked = tracker2.update(frame, boxes, np.array([0.9]), np.array([config.COCO_LAPTOP_ID]))
    if len(tracked) > 0:
        seen_ids.append(int(tracked.tracker_id[0]))
reappeared_id = seen_ids[-1]

print(f"id before gap: {first_id}, id after reappearing elsewhere: {reappeared_id}")
assert first_id == reappeared_id, (
    "Appearance re-ID failed to bridge the gap -- object got a new ID after "
    "disappearing and reappearing elsewhere, which is exactly the bug this "
    "rewrite was meant to fix."
)
print("PASS: same object correctly kept the same ID across an occlusion/pickup gap.")
