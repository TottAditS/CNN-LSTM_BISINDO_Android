"""
Training/MViTv2/Config/mvit_config.py
=======================================
Konfigurasi terpusat untuk fine-tuning MViTv2 (baseline transformer, pembanding
CNNsa-BiLSTM). Beroperasi di raw RGB video clip (bukan keypoint) — lihat
Training/MViTv2/Scripts/mvit_dataset.py.

NUM_CLASSES diimport dari Training.CNNsaBiLSTM.Config.train_config supaya jumlah
kelas TIDAK dobel-didefinisikan di dua tempat (single source of truth).
RESULTS_DIR diimport dari Training.Config.shared_config (SAMA dengan yang dipakai
CNNsaBiLSTM & dibaca Training/Scripts/comparator.py).
"""

import sys
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


ROOT = find_project_root()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Training.Config.shared_config import RESULTS_DIR, create_results_dir
from Training.CNNsaBiLSTM.Config.train_config import NUM_CLASSES
# Split ratio & seed di-IMPORT (bukan didefinisikan ulang) dari FeatureExtraction/Config —
# ini yang dipakai FeatureExtraction/Scripts/sequence_builder.py untuk split CNNsa-BiLSTM.
# WAJIB sumber yang sama supaya stratified_video_split() menghasilkan video val/test yang
# identik untuk kedua model (lihat FeatureExtraction/Scripts/video_split.py).
from FeatureExtraction.Config.feature_config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED


# ─── PATH DATA (output PreProcessing — raw frames, BUKAN keypoint) ────────────
PROCESSED_DIR  = ROOT / "Datasets" / "WLBisindo_Processed"
FRAME_MANIFEST = PROCESSED_DIR / "frame_manifest.json"

# ─── PATH OUTPUT TRAINING ─────────────────────────────────────────────────────
TRAINING_OUTPUT_DIR = ROOT / "Training" / "Output" / "mvit_v2"
CHECKPOINT_DIR       = TRAINING_OUTPUT_DIR / "checkpoints"
LOG_DIR              = TRAINING_OUTPUT_DIR / "logs"
# RESULTS_DIR sengaja TIDAK di-namespace ulang di sini — tetap shared (lihat docstring).


# ─── CLIP SETTINGS (harus konsisten dengan spesifikasi torchvision mvit_v2_s) ─
CLIP_LENGTH = 16          # frame per clip (minimum yang didukung mvit_v2_s)
FRAME_SIZE  = (224, 224)  # sudah sesuai output PreProcessing, tidak perlu resize ulang
KINETICS_MEAN = [0.45, 0.45, 0.45]
KINETICS_STD  = [0.225, 0.225, 0.225]


# ─── ARSITEKTUR & FREEZE STRATEGY ─────────────────────────────────────────────
# mvit_v2_s (torchvision, pretrained Kinetics-400) punya 16 MultiscaleBlock.
# FREEZE_UNTIL_BLOCK=12 -> blocks[0:12] dibekukan (fitur generik dari Kinetics),
# blocks[12:16] + norm + head baru (32 kelas) di-fine-tune. Dataset training di
# sini kecil (~1000 klip) & VRAM terbatas (8GB) — freeze sebagian besar backbone
# lebih tahan overfit & lebih cepat. Turunkan nilai ini (mis. 8 atau 0) kalau mau
# fine-tune lebih dalam / full fine-tune.
FREEZE_UNTIL_BLOCK = 12

MODEL_CONFIG = {
    "num_classes"       : NUM_CLASSES,
    "freeze_until_block": FREEZE_UNTIL_BLOCK,
    "pretrained"        : True,
    "dropout"           : 0.5,  # dipakai head baru, sama seperti default torchvision
}


# ─── HYPERPARAMETER TRAINING ──────────────────────────────────────────────────
TRAIN_CONFIG = {
    "batch_size"    : 4,      # kecil karena VRAM 8GB + clip 16x224x224; naikkan kalau muat
    "max_epochs"    : 30,     # lebih sedikit dari CNNsa-BiLSTM: per-epoch jauh lebih berat
    "backbone_lr"   : 1e-5,   # LR kecil untuk blocks yang di-fine-tune (sudah pretrained)
    "head_lr"       : 1e-3,   # LR lebih besar untuk head baru (random init)
    "weight_decay"  : 1e-4,
    "random_seed"   : 42,
    "use_amp"       : True,   # mixed precision — wajib untuk muat di VRAM 8GB
}


# ─── EARLY STOPPING ───────────────────────────────────────────────────────────
EARLY_STOPPING_CONFIG = {
    "monitor"   : "val_accuracy",
    "patience"  : 7,
    "min_delta" : 1e-4,
    "mode"      : "max",
}


# ─── LEARNING RATE SCHEDULER ──────────────────────────────────────────────────
LR_SCHEDULER_CONFIG = {
    "type"      : "ReduceLROnPlateau",
    "monitor"   : "val_loss",
    "factor"    : 0.5,
    "patience"  : 3,
    "min_lr"    : 1e-7,
}


# ─── CHECKPOINT ───────────────────────────────────────────────────────────────
CHECKPOINT_CONFIG = {
    "monitor"       : "val_accuracy",
    "save_best_only": True,
    "filename"      : "mvit_v2_best.pt",
}


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def create_output_dirs():
    """Buat semua direktori output MViTv2 jika belum ada (termasuk RESULTS_DIR shared)."""
    for d in [TRAINING_OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    create_results_dir()
    print(f"[MViTConfig] ROOT          : {ROOT}")
    print(f"[MViTConfig] Frame Manifest: {FRAME_MANIFEST}")
    print(f"[MViTConfig] Checkpoint Dir: {CHECKPOINT_DIR}")
    print(f"[MViTConfig] Log Dir       : {LOG_DIR}")
    print(f"[MViTConfig] Results Dir   : {RESULTS_DIR} (shared)")


def verify_data_exists() -> bool:
    """Cek apakah frame_manifest.json (output PreProcessing) sudah tersedia."""
    if not FRAME_MANIFEST.exists():
        print("[MViTConfig] ERROR: frame_manifest.json tidak ditemukan!")
        print(f"  {FRAME_MANIFEST}")
        print("  Jalankan PreProcessing/PreProcess.ipynb terlebih dahulu.")
        return False
    print("[MViTConfig] ✅ frame_manifest.json ditemukan.")
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
