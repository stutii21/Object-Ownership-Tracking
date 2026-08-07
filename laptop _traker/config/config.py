"""
Central configuration for the Laptop Ownership & Handover Tracking pipeline.
Tune these values to your specific camera / room before running on real footage.
"""

# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
# Which detector backend to use: "yolo" (Ultralytics, works anywhere) or
# "rfdetr" (Roboflow RF-DETR, transformer-based, fewer duplicate boxes in
# dense scenes, better under partial occlusion — but its pretrained weights
# download from storage.googleapis.com on first use, which some locked-down
# sandboxes/networks block. Use "yolo" if that domain isn't reachable for you.
DETECTOR_BACKEND = "yolo"  # "yolo" | "rfdetr"

# RF-DETR variant, only used when DETECTOR_BACKEND == "rfdetr".
# nano (fastest) < small < medium < base (balanced default) < large (most accurate)
RFDETR_VARIANT = "base"

# Pretrained YOLO weights (COCO classes). "yolo11n.pt" = fastest, "yolo11s.pt"/"yolo11m.pt"
# = more accurate but slower. Ultralytics downloads the weight file automatically on first run.
MODEL_WEIGHTS = "yolo11m.pt"
# Inference resolution passed to YOLO. Lower = faster but can miss small objects;
# higher = slower but better recall. 480 was empirically the best speed/recall
# tradeoff on a steep overhead camera angle where the laptop appears small.
INFERENCE_IMGSZ = 480

# COCO class ids we care about (standard COCO, unchanged in Ultralytics' pretrained models)
COCO_PERSON_ID = 0
COCO_LAPTOP_ID = 63

DETECT_CLASSES = [COCO_PERSON_ID, COCO_LAPTOP_ID]
# NOTE: a steep overhead/oblique camera angle is a hard case for a laptop
# detector trained mostly on frontal desk photos. 0.15 trades some false
# positives for much better recall; tighten this back up if you see false
# laptop detections on your footage.
CONF_THRESHOLD = 0.15
IOU_THRESHOLD_NMS = 0.5

# Optional: region of interest as a normalized polygon [(x, y), ...] in 0..1 coords,
# relative to frame width/height. Set to None to use the whole frame.
ROI_POLYGON = None

# ---------------------------------------------------------------------------
# Tracker (ByteTrack via `supervision`)
# ---------------------------------------------------------------------------
TRACK_ACTIVATION_THRESHOLD = 0.25
# Frames to keep a lost track alive before dropping it and assigning a new ID
# on reappearance. Raised from the default 30 because the laptop detector
# above has real gaps (occlusion + a hard camera angle) longer than 1 second.
LOST_TRACK_BUFFER = 90
MINIMUM_MATCHING_THRESHOLD = 0.8
FRAME_RATE = 30                 # will be overridden by the actual video's fps at runtime

# ---------------------------------------------------------------------------
# Ownership assignment
# ---------------------------------------------------------------------------
# score = W_DIST * (1 / (1 + normalized_distance))
#       + W_IOU  * overlap_between_person_and_laptop_boxes
#       + W_TEMP * temporal_consistency_bonus (favors the current owner, to avoid flicker)
W_DIST = 0.5
W_IOU = 0.3
W_TEMP = 0.2

# Distance is normalized by the frame diagonal so the weight is resolution-independent.
# Minimum score required before we assign ANY owner (otherwise laptop is "unattended").
MIN_OWNERSHIP_SCORE = 0.15

# ---------------------------------------------------------------------------
# Handover detection
# ---------------------------------------------------------------------------
# If you know there is only ever ONE instance of the tracked object (e.g. one
# laptop) in the scene, set this to True. It collapses whatever tracker ID
# ByteTrack assigns the object to a single canonical ID, so a track reset
# (which happens when the object is picked up, moved, and reappears in a very
# different position than where it was lost -- IoU-based matching can't
# bridge that) doesn't look like a "new laptop" to the ownership/handover
# logic. If you have MULTIPLE laptops in frame simultaneously, set this False.
SINGLE_OBJECT_MODE = True
CANONICAL_OBJECT_ID = 1000

# ---------------------------------------------------------------------------
# Appearance re-identification (src/reid.py, used by src/tracker.py's HybridTracker)
# ---------------------------------------------------------------------------
# Cosine similarity (0-1) two HSV histograms must reach to be considered the
# same physical object across a track gap. Lower = more permissive (bridges
# bigger appearance changes from lighting/angle, but risks merging two
# different but similar-colored objects). 0.75 is a reasonable starting point;
# tighten it if you see two different laptops merged into one ID.
REID_SIMILARITY_THRESHOLD = 0.75
# How many frames a lost track's appearance is remembered for potential
# re-matching. Should be >= LOST_TRACK_BUFFER's practical gap length.
REID_MEMORY_FRAMES = 90

# A candidate new owner must win the ownership score for this many consecutive
# frames before we confirm a handover. This prevents single-frame detection noise
# from generating spurious handover events.
HANDOVER_CONFIRMATION_FRAMES = 15

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
OUTPUT_VIDEO_CODEC = "mp4v"
EVENT_LOG_CSV = "outputs/events.csv"
EVENT_LOG_SQLITE = "outputs/events.db"
ANNOTATED_VIDEO_PATH = "outputs/annotated.mp4"
