# Laptop Ownership & Handover Tracking

A working pipeline for tracking which person "owns" which laptop in a video,
and logging every handover between people over time. Built from:

**Detector (RF-DETR backed YOLO11, pretrained on COCO)** → **Tracker (ByteTrack)** → **Ownership
assignment (proximity + overlap + temporal consistency)** → **Handover
detection (debounced)** → **CSV/SQLite event log + annotated video +
Streamlit dashboard**.

## Why no custom training is needed (for this version)

Both `person` and `laptop` already exist as classes in the COCO dataset,
which the pretrained YOLO weights were trained on. That means you can run
this **today, on your existing mp4, with zero annotation or training**.

If you later need a class that ISN'T in COCO (e.g. a specific badge, ID
card, or a conveyor-belt part), you would need to:
1. Annotate a few hundred frames in CVAT (export as COCO JSON)
2. Fine-tune on that data 
3. Point `config.MODEL_WEIGHTS` at your fine-tuned `.pt` file

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The first run will auto-download the YOLO11 nano weights (~5MB) from
Ultralytics. If you have a GPU, install the CUDA build of PyTorch first for
much faster inference.

## Run on your video

```bash
python main.py --input /path/to/your_video.mp4 --output outputs/annotated.mp4
```

This produces:
- `outputs/annotated.mp4` — boxes + tracker IDs + "Owner: X" labels + a red
  line from each laptop to its current owner
- `outputs/events.csv` — every confirmed init/handover/unattended event
- `outputs/events.db` — same data in SQLite (used by the dashboard)

Console output streams each confirmed event as it happens, e.g.:
```
[ 12.40s] frame   372 | laptop 3 | HANDOVER: 1 -> 2 (conf=0.71)
```

For a quick smoke test on just the first few seconds:
```bash
python main.py --input your_video.mp4 --max-frames 300
```

## Review results

```bash
streamlit run dashboard.py
```
Shows the event table, a per-laptop ownership chain (e.g. `1 → 2 → 3`), and
the annotated video.

## How ownership is decided

For every (person, laptop) pair, each frame:

```
score = 0.5 * proximity_term + 0.3 * overlap_term + 0.2 * temporal_consistency_term
```

- **proximity_term**: closer centroid distance → higher score (normalized by
  frame diagonal, so it works at any resolution)
- **overlap_term**: IoU between an expanded "reach zone" around the person
  and the laptop box — rewards someone actually leaning over/using it, not
  just standing nearby
- **temporal_consistency_term**: a bonus for whoever is already the
  confirmed owner, so momentary occlusion or someone walking past doesn't
  cause ownership to flicker

The person with the highest score above `MIN_OWNERSHIP_SCORE` wins that
frame. A new winner only gets **confirmed** (and logged) after holding the
lead for `HANDOVER_CONFIRMATION_FRAMES` (default 15) consecutive frames —
this is what prevents single-frame detector noise from generating fake
handover events.

All of these are tunable in `config/config.py`.

## Where to put your video

Your video goes in `data/` — e.g. `data/video_slow.mp4`. This zip already
has a real video and a working `data/boxes.json` in it as a worked example
(see below), so you can see exactly how it's structured before swapping in
your own file.

```bash
python select_and_track.py --input data/video_slow.mp4
```
(interactive — drag boxes yourself, see the section above)

or, if you already know the pixel coordinates (e.g. from makesense.ai, or
you want to skip the GUI entirely):
```bash
python select_and_track.py --input data/video_slow.mp4 --boxes-file data/boxes.json
```

`data/boxes.json` in this zip was actually run end-to-end (not just
written) against `data/video_slow.mp4` — output: `outputs/events.csv`
logged one `init` event at t=5.16s assigning the laptop to `person_A`,
matching every other method tried in this project. Open it to see the
exact format expected, then edit the coordinates/labels for your own clip.

## Detector backend: YOLO vs RF-DETR

`config.DETECTOR_BACKEND` selects the detector (`src/detector_rfdetr.py`'s
`build_detector()` factory picks the right one — main.py doesn't need to
know which is active):

- `"yolo"`  — Ultralytics YOLO11. 
- `"rfdetr"` — Roboflow's RF-DETR, transformer-based. Generally handles
  crowded/occluded scenes better than a CNN detector because self-attention
  reasons about the whole image jointly rather than purely local regions —
  directly relevant to your footage (6-10 people in frame at once).

this repo's sandbox: 595 frames of your video in 228.6s (2.6 fps) on CPU.
**Important limitation, stated plainly:** RF-DETR downloads its pretrained
weights from `storage.googleapis.com` the first time each variant is
instantiated.

## Tracking: what "best way to identify tracking" means here, and what was built

Plain ByteTrack (motion + IoU matching, frame-to-frame) breaks in exactly
the way your footage stresses it: the laptop's detection confidence is low
and intermittent (measured: 0.15-0.79, frequently invisible for over a
second at a time), so when it reappears, its position often no longer
overlaps where it was last seen -- ByteTrack then assigns a **new** ID,
and naive downstream logic sees a "new" laptop rather than a handover.

This repo's `src/tracker.py` (`HybridTracker`) adds appearance-based
re-identification on top of ByteTrack (`src/reid.py`): every tracked box
gets a compact HSV color-histogram signature; when ByteTrack mints a new
ID, its signature is compared against recently-lost tracks of the same
class within a time/position window, and the OLD id is re-attached if the
appearance genuinely matches. This is the same core idea behind
DeepSORT/BoT-SORT's learned appearance embedding, scaled down to run on
CPU with no extra model download (so it isn't blocked by the same
network restriction as RF-DETR).

**This was tested, not just described** — `tests/test_pipeline_logic.py`
includes a gap-bridging test: an object disappears for 40 frames then
reappears at a completely different position with the same color; the
hybrid tracker correctly re-attaches the original ID (plain ByteTrack
would not). Both tests in that file pass.

If you later need to scale to many visually-similar objects of the same
class (e.g. 10 identical laptops), color histograms won't discriminate
well enough — at that point upgrade to a learned embedding via BoT-SORT or
StrongSORT (e.g. the `boxmot` package), which follows the same pattern but
needs its own downloaded ReID weights.

## Zone-based ownership (main_zones.py) — fast to try, real result found

Since your seats measured as stable across the whole clip (person centroids
varied by only 5-23px over 12 sampled frames — see `tools/find_seat_zones.py`
output), this approach skips person detection/tracking entirely: only the
laptop is detected each frame, and "owner" = whichever fixed seat anchor its
centroid is nearest to. No person-tracking error can leak into the ownership
decision at all, and re-deriving anchors for a new camera takes one command:

```bash
python tools/find_seat_zones.py --input your_video.mp4
python main_zones.py --input your_video.mp4
```

**What actually happened when this ran on your real video — an important,
honest finding:** it confirmed TWO handovers (Seat_D→Seat_C at 11.49s,
Seat_C→Seat_B at 12.22s) in exactly the borderline window the
proximity-based method (main.py) had flagged but not confirmed. Promising —
until visual inspection of `outputs/zones_annotated.mp4` at those
timestamps showed **this is a false positive**: it's the same person (the
one closest to camera) holding and gesturing with the laptop on his lap the
whole time, not a handover to someone else. His arm movement swung the
laptop's on-screen position across two zone boundaries.

**The real lesson:** zone-based ownership works well when an object rests
in a stable position tied to whoever's "seat" it's in (e.g. sits on a desk).
It works poorly when the object is actively hand-held and gestured with,
because "nearest fixed point" isn't a good proxy for "whose hands is it in"
once hands move independently of seat position. For hand-held objects,
either (a) trust the proximity+temporal-consistency method in main.py,
which correctly rejected this exact window as noise, or (b) go further and
track actual hand/wrist keypoints (e.g. MediaPipe Hands) rather than
seat-proximity or person-bounding-box proximity, which would distinguish
"still in this person's hand" from "now in a different person's hand" directly.

## Tuning for your specific footage

| Symptom | Likely fix |
|---|---|
| Ownership flickers between two people standing close together | Increase `HANDOVER_CONFIRMATION_FRAMES`, increase `W_TEMP` |
| Real handovers are detected too slowly | Decrease `HANDOVER_CONFIRMATION_FRAMES` |
| Laptop assigned to someone across the room | Increase `W_DIST`, decrease the `5.0` distance-falloff multiplier in `ownership.py` for a gentler curve, or vice versa |
| False positives from background objects | Set `ROI_POLYGON` in config to mask out irrelevant areas |
| Wrong camera angle / severe fisheye distortion | Undistort frames before detection (add a calibration step — not included here) |



## Extending to the conveyor-belt / multi-observer variant

The architecture is identical — swap "laptop" for "conveyor object" and
"laptop owner" for "current observer". The one change needed: "conveyor
object" is not a COCO class, so you'd need to annotate ~200-500 frames in
CVAT and fine-tune YOLO (or RF-DETR, if you want the transformer-based
detector originally discussed) on that one class before this pipeline will
detect it. Everything downstream (tracker, assignment, handover, logging,
dashboard) works unchanged — the assignment formula in `ownership.py` is
already generic (`person` × `object`), not laptop-specific.

## Project structure

```
laptop_tracker/
├── config/config.py       # all tunable thresholds/weights
├── src/
│   ├── detector.py         # wrapper (person + laptop)
│   ├── tracker.py          # ByteTrack wrapper
│   ├── ownership.py        # scoring algorithm
│   ├── handover.py         # debounced event confirmation
│   └── event_logger.py     # CSV + SQLite logging
├── main.py                  # pipeline entry point
├── dashboard.py              # Streamlit review UI
├── tests/test_pipeline_logic.py  # logic test (passing, no real video needed)
└── requirements.txt
```
