"""
Feature extraction configuration: paths, MediaPipe settings, sequence length,
data split, and keypoint augmentation.
"""

from pathlib import Path


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "Datasets").exists() and (parent / "PreProcessing").exists():
            return parent
    raise RuntimeError(
        "Project root not found. "
        "The root must contain the Datasets/ and PreProcessing/ folders."
    )


# --- Paths --------------------------------------------------------------------
ROOT = find_project_root()

DATASET_DIR         = ROOT / "Datasets" / "WLBisindo"
PROCESSED_DIR       = ROOT / "Datasets" / "WLBisindo_Processed"
FEATURE_OUTPUT_DIR  = ROOT / "Datasets" / "WLBisindo_Features"

KEYPOINT_DIR        = FEATURE_OUTPUT_DIR / "keypoints"    # one .npy per video
SEQUENCE_DIR        = FEATURE_OUTPUT_DIR / "sequences"    # final .npz files


# --- Dataset ------------------------------------------------------------------
# WL-BISINDO: signer0-signer4, label0-label31. signer0 recorded 12 of the 32
# labels, and each (signer, label) pair has 10 or 15 samples.
NUM_SIGNERS                  = 5
NUM_LABELS                   = 32
SAMPLES_PER_LABEL_PER_SIGNER = None  # varies, not used to enumerate files


# --- Frames (must match PreProcessing) ----------------------------------------
FRAME_SIZE       = (224, 224)
MAX_FRAMES       = 60
SEQUENCE_LENGTH  = 30


# --- MediaPipe ----------------------------------------------------------------
NUM_HANDS           = 2
KEYPOINTS_PER_HAND  = 21
COORDS_PER_KEYPOINT = 3
FEATURES_PER_FRAME  = NUM_HANDS * KEYPOINTS_PER_HAND * COORDS_PER_KEYPOINT  # 126

MP_STATIC_IMAGE_MODE        = False
MP_MAX_NUM_HANDS            = NUM_HANDS
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE  = 0.5


# --- Keypoint normalization ---------------------------------------------------
# Subtract the wrist and divide by the wrist-to-middle-finger-MCP distance, so the
# features do not depend on hand position or size.
NORMALIZE_KEYPOINTS = True


# --- Sequences ----------------------------------------------------------------
PAD_VALUE            = 0.0
CENTER_CROP_TEMPORAL = True   # True = keep the middle frames, False = keep the first frames


# --- Split --------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42


# --- Augmentation (keypoint level, training split only) -----------------------
# The model reads normalized keypoints, so photometric augmentation would have no
# effect; only geometric, temporal, and detection-noise augmentations are used.
# The split is made before augmentation, so validation and test hold original data only.
AUGMENTATION_CONFIG = {
    "enabled"               : True,
    "techniques"            : ["horizontal_flip", "rotation", "zoom", "jitter", "temporal_speed", "hand_dropout"],
    "horizontal_flip_prob"  : 0.5,
    "rotation_range"        : (-20, 20),    # degrees
    "zoom_range"            : (0.8, 1.2),   # scale of each keypoint's distance from the wrist
    "jitter_std"            : 0.02,         # per-coordinate Gaussian noise, imitates MediaPipe detection noise
    "temporal_speed_range"  : (0.85, 1.15), # signing speed variation
    "hand_dropout_prob"     : 0.15,         # chance that a hand disappears for part of the sequence
    "hand_dropout_max_frac" : 0.3,          # maximum fraction of the sequence affected by one dropout
    "augment_multiplier"    : 4,            # 1 original + 4 augmented copies

    # Signer-Mixup: linear interpolation between two sequences of the same class
    # recorded by different signers, producing synthetic in-between signer styles.
    "signer_mixup_count"      : 2,          # mixup copies per training video
    "signer_mixup_alpha_range": (0.3, 0.7), # mixing weight, kept away from 0 and 1
}


# --- Misc ---------------------------------------------------------------------
VERBOSE = True

LABEL_MAP = {i: f"label_{i}" for i in range(NUM_LABELS)}


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
              f"({NUM_HANDS} hands x {KEYPOINTS_PER_HAND} kpts x {COORDS_PER_KEYPOINT} coords)")


if __name__ == "__main__":
    create_output_dirs()
    print("Config OK.")
