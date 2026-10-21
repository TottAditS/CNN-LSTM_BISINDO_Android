"""
CNNsa-BiLSTM training configuration: data paths, model hyperparameters,
optimizer, scheduler, and early stopping.

RESULTS_DIR comes from Training/Config/shared_config.py and is shared with MViTv2.
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


# --- Data (FeatureExtraction output) ------------------------------------------
SEQUENCE_DIR    = ROOT / "Datasets" / "WLBisindo_Features" / "sequences"
TRAIN_NPZ       = SEQUENCE_DIR / "train_sequences.npz"
VAL_NPZ         = SEQUENCE_DIR / "val_sequences.npz"
TEST_NPZ        = SEQUENCE_DIR / "test_sequences.npz"

# --- Training output (checkpoints and logs of this model only) ----------------
TRAINING_OUTPUT_DIR = ROOT / "Training" / "Output" / "cnnsa_bilstm"
CHECKPOINT_DIR      = TRAINING_OUTPUT_DIR / "checkpoints"
LOG_DIR             = TRAINING_OUTPUT_DIR / "logs"


# --- Input dimensions (must match FeatureExtraction) --------------------------
SEQUENCE_LENGTH  = 30     # frames per sequence
FEATURES_PER_FRAME = 126  # 2 hands x 21 landmarks x 3 coordinates
NUM_CLASSES      = 32     # label0-label31


# --- Model --------------------------------------------------------------------
MODEL_CONFIG = {
    "input_size"    : FEATURES_PER_FRAME,   # 126
    "feature_dim"   : 256,    # per-frame CNN output size
    "hidden_size"   : 256,    # BiLSTM hidden size per direction (512 bidirectional)
    "num_layers"    : 2,      # BiLSTM layers
    "num_classes"   : NUM_CLASSES,
    "dropout"       : 0.3,
    "attn_reduction": 16,     # reduction ratio of the channel attention
}


# --- Training -----------------------------------------------------------------
TRAIN_CONFIG = {
    "batch_size"    : 64,
    "max_epochs"    : 100,
    "learning_rate" : 1e-3,
    "weight_decay"  : 1e-4,
    "random_seed"   : 42,
}


EARLY_STOPPING_CONFIG = {
    "monitor"   : "val_accuracy",
    "patience"  : 10,               # epochs without improvement before stopping
    "min_delta" : 1e-4,             # minimum improvement
    "mode"      : "max",
}


LR_SCHEDULER_CONFIG = {
    "type"      : "ReduceLROnPlateau",
    "monitor"   : "val_loss",
    "factor"    : 0.5,      # LR multiplier on plateau
    "patience"  : 5,        # epochs before reducing the LR
    "min_lr"    : 1e-6,
}


CHECKPOINT_CONFIG = {
    "monitor"       : "val_accuracy",
    "save_best_only": True,
    "filename"      : "cnnsa_bilstm_best.pt",
}


def create_output_dirs():
    """Create the training output folders."""
    for d in [TRAINING_OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[TrainConfig] ROOT          : {ROOT}")
    print(f"[TrainConfig] Sequence Dir  : {SEQUENCE_DIR}")
    print(f"[TrainConfig] Checkpoint Dir: {CHECKPOINT_DIR}")
    print(f"[TrainConfig] Log Dir       : {LOG_DIR}")


def verify_data_exists() -> bool:
    """Check that the FeatureExtraction .npz files exist."""
    missing = []
    for name, path in [("train", TRAIN_NPZ), ("val", VAL_NPZ), ("test", TEST_NPZ)]:
        if not path.exists():
            missing.append(f"  [MISSING] {name}: {path}")
    if missing:
        print("[TrainConfig] ERROR: sequence files not found.")
        print("  Run FeatureExtraction/FeatureExtraction.ipynb first.\n")
        for m in missing:
            print(m)
        return False
    print("[TrainConfig] All sequence files found.")
    return True


if __name__ == "__main__":
    create_output_dirs()
    verify_data_exists()
    print("\nModel config:")
    for k, v in MODEL_CONFIG.items():
        print(f"  {k:<16}: {v}")
    print("\nTrain config:")
    for k, v in TRAIN_CONFIG.items():
        print(f"  {k:<16}: {v}")
