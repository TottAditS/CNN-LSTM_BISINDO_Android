"""
PyTorch to ONNX export settings for both models.

Checkpoint paths are imported from each model's training config, so the export
always reads the checkpoint the trainer wrote.
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

from Training.CNNsaBiLSTM.Config.train_config import (
    CHECKPOINT_DIR as CNNSA_CHECKPOINT_DIR,
    CHECKPOINT_CONFIG as CNNSA_CHECKPOINT_CONFIG,
    SEQUENCE_LENGTH, FEATURES_PER_FRAME,
)
from Training.MViTv2.Config.mvit_config import (
    CHECKPOINT_DIR as MVIT_CHECKPOINT_DIR,
    CHECKPOINT_CONFIG as MVIT_CHECKPOINT_CONFIG,
    CLIP_LENGTH, FRAME_SIZE,
)

CNNSA_CHECKPOINT_PATH = CNNSA_CHECKPOINT_DIR / CNNSA_CHECKPOINT_CONFIG["filename"]
MVIT_CHECKPOINT_PATH  = MVIT_CHECKPOINT_DIR / MVIT_CHECKPOINT_CONFIG["filename"]

# --- ONNX output --------------------------------------------------------------
EXPORT_OUTPUT_DIR = ROOT / "Export" / "Output"
ONNX_DIR           = EXPORT_OUTPUT_DIR / "onnx"
EXPORT_CHARTS_DIR  = EXPORT_OUTPUT_DIR / "charts"

CNNSA_ONNX_PATH = ONNX_DIR / "cnnsa_bilstm.onnx"
MVIT_ONNX_PATH  = ONNX_DIR / "mvit_v2.onnx"

# --- Exporter settings --------------------------------------------------------
# CNNsa-BiLSTM uses the legacy TorchScript exporter (dynamo=False).
# MViTv2 needs the dynamo exporter (dynamo=True, requires onnxscript); the legacy
# exporter produces a graph that fails in ONNX Runtime. The dynamo exporter
# requires opset 18 or higher.
CNNSA_OPSET = 17
CNNSA_USE_DYNAMO = False

MVIT_OPSET = 18
MVIT_USE_DYNAMO = True

# Fixed batch size of 1: the app classifies one sequence or clip at a time, and a
# fixed batch avoids PyTorch's LSTM dynamic-batch warning.
BATCH_SIZE = 1

# Dummy input shapes for tracing (must match the training configs).
CNNSA_INPUT_SHAPE = (BATCH_SIZE, SEQUENCE_LENGTH, FEATURES_PER_FRAME)          # (1, 30, 126)
MVIT_INPUT_SHAPE  = (BATCH_SIZE, 3, CLIP_LENGTH, FRAME_SIZE[0], FRAME_SIZE[1])  # (1, 3, 16, 224, 224)


def create_output_dirs():
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[ExportConfig] ROOT           : {ROOT}")
    print(f"[ExportConfig] ONNX Output Dir: {ONNX_DIR}")


def verify_checkpoints_exist() -> bool:
    """Check that both training checkpoints exist."""
    missing = []
    for name, path in [("CNNsa-BiLSTM", CNNSA_CHECKPOINT_PATH), ("MViTv2", MVIT_CHECKPOINT_PATH)]:
        if not path.exists():
            missing.append(f"  [MISSING] {name}: {path}")
    if missing:
        print("[ExportConfig] ERROR: checkpoints not found.")
        print("  Run Training/CNNsaBiLSTM/Train.ipynb and Training/MViTv2/TrainMViT.ipynb first.\n")
        for m in missing:
            print(m)
        return False
    print("[ExportConfig] All checkpoints found.")
    return True


if __name__ == "__main__":
    create_output_dirs()
    verify_checkpoints_exist()
    print("CNNsa-BiLSTM checkpoint:", CNNSA_CHECKPOINT_PATH)
    print("MViTv2 checkpoint      :", MVIT_CHECKPOINT_PATH)
