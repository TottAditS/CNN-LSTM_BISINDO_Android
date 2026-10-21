"""
MViTv2 fine-tuning configuration. MViTv2 reads RGB clips, not keypoints
(see Training/MViTv2/Scripts/mvit_dataset.py).

NUM_CLASSES comes from the CNNsa-BiLSTM config, the split ratios and seed from
the FeatureExtraction config, and RESULTS_DIR from the shared config, so both
models use the same classes, the same split, and the same results folder.
"""

import sys
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


ROOT = find_project_root()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Training.Config.shared_config import RESULTS_DIR, create_results_dir
from Training.CNNsaBiLSTM.Config.train_config import NUM_CLASSES
# Same source as sequence_builder.py, so stratified_video_split() gives both models
# identical val/test videos.
from FeatureExtraction.Config.feature_config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED


# --- Data (PreProcessing output: RGB frames) ----------------------------------
PROCESSED_DIR  = ROOT / "Datasets" / "WLBisindo_Processed"
FRAME_MANIFEST = PROCESSED_DIR / "frame_manifest.json"

# --- Training output ----------------------------------------------------------
TRAINING_OUTPUT_DIR = ROOT / "Training" / "Output" / "mvit_v2"
CHECKPOINT_DIR       = TRAINING_OUTPUT_DIR / "checkpoints"
LOG_DIR              = TRAINING_OUTPUT_DIR / "logs"


# --- Clips (torchvision mvit_v2_s input) --------------------------------------
CLIP_LENGTH = 16          # frames per clip
FRAME_SIZE  = (224, 224)  # matches the PreProcessing output
KINETICS_MEAN = [0.45, 0.45, 0.45]
KINETICS_STD  = [0.225, 0.225, 0.225]


# --- Model and freezing -------------------------------------------------------
# mvit_v2_s has 16 MultiscaleBlocks. With FREEZE_UNTIL_BLOCK = 12, blocks[0:12]
# keep their Kinetics-400 weights and blocks[12:16], the final norm, and the new
# 32-class head are fine-tuned. Lower the value for deeper fine-tuning.
FREEZE_UNTIL_BLOCK = 12

MODEL_CONFIG = {
    "num_classes"       : NUM_CLASSES,
    "freeze_until_block": FREEZE_UNTIL_BLOCK,
    "pretrained"        : True,
    "dropout"           : 0.5,  # new head, same as the torchvision default
}


# --- Training -----------------------------------------------------------------
TRAIN_CONFIG = {
    "batch_size"    : 4,      # 16x224x224 clips on an 8 GB GPU
    "max_epochs"    : 30,
    "backbone_lr"   : 1e-5,   # pretrained blocks
    "head_lr"       : 1e-3,   # new head
    "weight_decay"  : 1e-4,
    "random_seed"   : 42,
    "use_amp"       : True,   # mixed precision, needed to fit in 8 GB
}


EARLY_STOPPING_CONFIG = {
    "monitor"   : "val_accuracy",
    "patience"  : 7,
    "min_delta" : 1e-4,
    "mode"      : "max",
}


LR_SCHEDULER_CONFIG = {
    "type"      : "ReduceLROnPlateau",
    "monitor"   : "val_loss",
    "factor"    : 0.5,
    "patience"  : 3,
    "min_lr"    : 1e-7,
}


CHECKPOINT_CONFIG = {
    "monitor"       : "val_accuracy",
    "save_best_only": True,
    "filename"      : "mvit_v2_best.pt",
}


def create_output_dirs():
    """Create the MViTv2 output folders and the shared results folder."""
    for d in [TRAINING_OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    create_results_dir()
    print(f"[MViTConfig] ROOT          : {ROOT}")
    print(f"[MViTConfig] Frame Manifest: {FRAME_MANIFEST}")
    print(f"[MViTConfig] Checkpoint Dir: {CHECKPOINT_DIR}")
    print(f"[MViTConfig] Log Dir       : {LOG_DIR}")
    print(f"[MViTConfig] Results Dir   : {RESULTS_DIR} (shared)")


def verify_data_exists() -> bool:
    """Check that frame_manifest.json from PreProcessing exists."""
    if not FRAME_MANIFEST.exists():
        print("[MViTConfig] ERROR: frame_manifest.json not found.")
        print(f"  {FRAME_MANIFEST}")
        print("  Run PreProcessing/PreProcess.ipynb first.")
        return False
    print("[MViTConfig] frame_manifest.json found.")
    return True


if __name__ == "__main__":
    create_output_dirs()
    verify_data_exists()
    print("\nModel config:")
    for k, v in MODEL_CONFIG.items():
        print(f"  {k:<18}: {v}")
    print("\nTrain config:")
    for k, v in TRAIN_CONFIG.items():
        print(f"  {k:<18}: {v}")
