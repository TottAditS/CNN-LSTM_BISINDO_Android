"""
FeatureExtraction/Config/feature_config.py
Auto-detect ROOT dan konfigurasi semua setting untuk feature extraction.
"""

import os
from pathlib import Path


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "Datasets").exists() and (parent / "PreProcessing").exists():
            return parent
    raise RuntimeError(
        "Tidak bisa menemukan root project. "
        "Pastikan direktori Datasets/ dan PreProcessing/ ada di root."
    )


# ─── ROOT & PATH ──────────────────────────────────────────────────────────────
ROOT = find_project_root()

DATASET_DIR         = ROOT / "Datasets" / "WLBisindo"
PROCESSED_DIR       = ROOT / "Datasets" / "WLBisindo_Processed"
FEATURE_OUTPUT_DIR  = ROOT / "Datasets" / "WLBisindo_Features"

KEYPOINT_DIR        = FEATURE_OUTPUT_DIR / "keypoints"    # per-video .npy
SEQUENCE_DIR        = FEATURE_OUTPUT_DIR / "sequences"    # final .npz


# ─── DATASET INFO ─────────────────────────────────────────────────────────────
NUM_SIGNERS                  = 4
NUM_LABELS                   = 40
SAMPLES_PER_LABEL_PER_SIGNER = 10


# ─── FRAME SETTINGS (harus sama dengan PreProcessing) ─────────────────────────
FRAME_SIZE       = (224, 224)
MAX_FRAMES       = 60
SEQUENCE_LENGTH  = 30


# ─── MEDIAPIPE SETTINGS ───────────────────────────────────────────────────────
NUM_HANDS           = 2
KEYPOINTS_PER_HAND  = 21
COORDS_PER_KEYPOINT = 3
FEATURES_PER_FRAME  = NUM_HANDS * KEYPOINTS_PER_HAND * COORDS_PER_KEYPOINT  # 126

MP_STATIC_IMAGE_MODE        = False
MP_MAX_NUM_HANDS            = NUM_HANDS
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE  = 0.5


# ─── NORMALISASI KEYPOINT ─────────────────────────────────────────────────────
# Subtract wrist, scale by wrist→MCP9 distance → invariant terhadap posisi & ukuran tangan
NORMALIZE_KEYPOINTS = True


# ─── SEQUENCE SETTINGS ────────────────────────────────────────────────────────
PAD_VALUE            = 0.0
CENTER_CROP_TEMPORAL = True   # True = ambil frame tengah; False = ambil N frame pertama


# ─── SPLIT SETTINGS ───────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42


# ─── PROCESSING SETTINGS ──────────────────────────────────────────────────────
VERBOSE = True


# ─── LABEL MAP ────────────────────────────────────────────────────────────────
LABEL_MAP = {i: f"label_{i}" for i in range(NUM_LABELS)}


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def get_keypoint_path(signer_id: int, label_id: int, sample_id: int) -> Path:
    fname = f"signer{signer_id}_label{label_id}_sample{sample_id}.npy"
    return KEYPOINT_DIR / fname


def get_sequence_output_path(split: str) -> Path:
    return SEQUENCE_DIR / f"{split}_sequences.npz"


def create_output_dirs():
    for d in [FEATURE_OUTPUT_DIR, KEYPOINT_DIR, SEQUENCE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    if VERBOSE:
        print(f"[Config] ROOT          : {ROOT}")
        print(f"[Config] Dataset       : {DATASET_DIR}")
        print(f"[Config] Keypoint Dir  : {KEYPOINT_DIR}")
        print(f"[Config] Sequence Dir  : {SEQUENCE_DIR}")
        print(f"[Config] Features/frame: {FEATURES_PER_FRAME} "
              f"({NUM_HANDS} hands × {KEYPOINTS_PER_HAND} kpts × {COORDS_PER_KEYPOINT} coords)")


if __name__ == "__main__":
    create_output_dirs()
    print("Config OK.")
