"""
Training/CNNsaBiLSTM/Config/train_config.py
============================================
Konfigurasi terpusat untuk training CNNsa-BiLSTM (proposed model).
- Path ke data (output FeatureExtraction)
- Hyperparameter model
- Setting training (optimizer, scheduler, early stopping)

Auto-detect ROOT menggunakan keberadaan folder Datasets/ dan PreProcessing/.
RESULTS_DIR (tempat evaluation_summary.json disimpan & dibaca comparator.py) berasal dari
Training/Config/shared_config.py -- dipakai bersama dengan model lain (mis. MViTv2).
"""

import sys
from pathlib import Path


# --- AUTO-DETECT ROOT ---------------------------------------------------------
def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "Datasets").exists() and (parent / "PreProcessing").exists():
            return parent
    raise RuntimeError(
        "Tidak bisa menemukan root project. "
        "Pastikan direktori Datasets/ dan PreProcessing/ ada di root."
    )


ROOT = find_project_root()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Training.Config.shared_config import RESULTS_DIR, create_results_dir


# --- PATH DATA (output dari FeatureExtraction) --------------------------------
SEQUENCE_DIR    = ROOT / "Datasets" / "WLBisindo_Features" / "sequences"
TRAIN_NPZ       = SEQUENCE_DIR / "train_sequences.npz"
VAL_NPZ         = SEQUENCE_DIR / "val_sequences.npz"
TEST_NPZ        = SEQUENCE_DIR / "test_sequences.npz"

# --- PATH OUTPUT TRAINING -----------------------------------------------------
# Checkpoint & log khusus CNNsaBiLSTM (namespaced, tidak bentrok dengan MViTv2).
# RESULTS_DIR tetap SHARED (lihat shared_config.py) supaya comparator.py bisa baca
# evaluation_summary.json dari kedua model di satu tempat yang sama.
TRAINING_OUTPUT_DIR = ROOT / "Training" / "Output" / "cnnsa_bilstm"
CHECKPOINT_DIR      = TRAINING_OUTPUT_DIR / "checkpoints"
LOG_DIR             = TRAINING_OUTPUT_DIR / "logs"


# --- DIMENSI INPUT (harus konsisten dengan FeatureExtraction) -----------------
SEQUENCE_LENGTH  = 30     # frame per sequence
FEATURES_PER_FRAME = 126  # 2 tangan x 21 keypoint x 3 coords
NUM_CLASSES      = 32     # jumlah kata BISINDO (sesuai label unik di Datasets/WLBisindo saat ini: label0-label31)


# --- ARSITEKTUR MODEL CNNsa-BiLSTM --------------------------------------------
MODEL_CONFIG = {
    "input_size"    : FEATURES_PER_FRAME,   # 126
    "feature_dim"   : 256,    # output ukuran per-frame CNN Block
    "hidden_size"   : 256,    # hidden size BiLSTM per arah (x2 = 512 bidirectional)
    "num_layers"    : 2,      # jumlah layer BiLSTM
    "num_classes"   : NUM_CLASSES,
    "dropout"       : 0.3,
    "attn_reduction": 16,     # reduction ratio untuk Spatial Attention
}


# --- HYPERPARAMETER TRAINING --------------------------------------------------
TRAIN_CONFIG = {
    "batch_size"    : 64,
    "max_epochs"    : 100,
    "learning_rate" : 1e-3,
    "weight_decay"  : 1e-4,
    "random_seed"   : 42,
}


# --- EARLY STOPPING -----------------------------------------------------------
EARLY_STOPPING_CONFIG = {
    "monitor"   : "val_accuracy",   # metrik yang dipantau
    "patience"  : 10,               # tunggu N epoch tanpa perbaikan sebelum stop
    "min_delta" : 1e-4,             # threshold minimal peningkatan
    "mode"      : "max",            # "max" karena kita ingin accuracy naik
}


# --- LEARNING RATE SCHEDULER --------------------------------------------------
LR_SCHEDULER_CONFIG = {
    "type"      : "ReduceLROnPlateau",
    "monitor"   : "val_loss",
    "factor"    : 0.5,      # kalikan LR dengan factor ini saat stuck
    "patience"  : 5,        # tunggu 5 epoch sebelum reduce LR
    "min_lr"    : 1e-6,
}


# --- CHECKPOINT ---------------------------------------------------------------
CHECKPOINT_CONFIG = {
    "monitor"       : "val_accuracy",
    "save_best_only": True,
    "filename"      : "cnnsa_bilstm_best.pt",
}


# --- HELPERS ------------------------------------------------------------------
def create_output_dirs():
    """Buat semua direktori output Training jika belum ada."""
    for d in [TRAINING_OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[TrainConfig] ROOT          : {ROOT}")
    print(f"[TrainConfig] Sequence Dir  : {SEQUENCE_DIR}")
    print(f"[TrainConfig] Checkpoint Dir: {CHECKPOINT_DIR}")
    print(f"[TrainConfig] Log Dir       : {LOG_DIR}")


def verify_data_exists() -> bool:
    """Cek apakah file .npz dari FeatureExtraction sudah tersedia."""
    missing = []
    for name, path in [("train", TRAIN_NPZ), ("val", VAL_NPZ), ("test", TEST_NPZ)]:
        if not path.exists():
            missing.append(f"  [MISSING] {name}: {path}")
    if missing:
        print("[TrainConfig] ERROR: File sequence tidak ditemukan!")
        print("  Jalankan FeatureExtraction/FeatureExtraction.ipynb terlebih dahulu.\n")
        for m in missing:
            print(m)
        return False
    print("[TrainConfig] Semua file sequence ditemukan.")
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
