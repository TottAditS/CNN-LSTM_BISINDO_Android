"""
Export/Config/export_config.py
================================
Konfigurasi export PyTorch -> ONNX untuk kedua model (CNNsa-BiLSTM & MViTv2).

Path checkpoint di-IMPORT (bukan didefinisikan ulang) dari Training/*/Config/ masing-masing
model -- single source of truth, supaya export selalu ambil checkpoint dari lokasi yang benar-
benar dipakai trainer/evaluator.
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

# --- OUTPUT ONNX ------------------------------------------------------------
EXPORT_OUTPUT_DIR = ROOT / "Export" / "Output"
ONNX_DIR           = EXPORT_OUTPUT_DIR / "onnx"

CNNSA_ONNX_PATH = ONNX_DIR / "cnnsa_bilstm.onnx"
MVIT_ONNX_PATH  = ONNX_DIR / "mvit_v2.onnx"

# --- EXPORT SETTINGS ----------------------------------------------------------
# Dua model butuh exporter ONNX yang BEDA -- dikonfirmasi lewat tes langsung, bukan tebakan:
#
# - CNNsa-BiLSTM: exporter LEGACY (TorchScript-based, dynamo=False) sudah cukup dan tervalidasi
#   (selisih vs PyTorch ~1e-6, ONNX Runtime bahkan lebih cepat dari PyTorch di CPU).
# - MViTv2: exporter legacy MENGHASILKAN model yang "sukses" di-export tapi GAGAL dijalankan di
#   ONNX Runtime ("Loop subgraph input 0 has unknown shape") -- pooling-attention MViT ke-trace
#   jadi Loop node dinamis yang shape-nya tidak bisa di-infer exporter lama. Exporter BARU
#   (torch.export-based, dynamo=True, butuh package `onnxscript`) berhasil menghasilkan model
#   yang benar-benar jalan di ONNX Runtime (selisih vs PyTorch ~1.25e-6). Exporter baru ini juga
#   otomatis naikkan opset ke >=18 kalau diminta opset lebih rendah -- makanya MVIT_OPSET=18.
CNNSA_OPSET = 17
CNNSA_USE_DYNAMO = False

MVIT_OPSET = 18
MVIT_USE_DYNAMO = True

# Batch size DIFIX ke 1 (bukan dynamic) -- sesuai use-case nyata: inferensi Android satu
# sequence/clip sekali jalan (bukan batch). Ini juga menghindari warning PyTorch soal LSTM
# + dynamic batch size (lihat CNNsa-BiLSTM export).
BATCH_SIZE = 1

# Shape dummy input untuk tracing export (harus konsisten dengan Training/*/Config).
CNNSA_INPUT_SHAPE = (BATCH_SIZE, SEQUENCE_LENGTH, FEATURES_PER_FRAME)          # (1, 30, 126)
MVIT_INPUT_SHAPE  = (BATCH_SIZE, 3, CLIP_LENGTH, FRAME_SIZE[0], FRAME_SIZE[1])  # (1, 3, 16, 224, 224)


def create_output_dirs():
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[ExportConfig] ROOT           : {ROOT}")
    print(f"[ExportConfig] ONNX Output Dir: {ONNX_DIR}")


def verify_checkpoints_exist() -> bool:
    """Cek apakah checkpoint hasil training (Training/CNNsaBiLSTM & Training/MViTv2) sudah ada."""
    missing = []
    for name, path in [("CNNsa-BiLSTM", CNNSA_CHECKPOINT_PATH), ("MViTv2", MVIT_CHECKPOINT_PATH)]:
        if not path.exists():
            missing.append(f"  [MISSING] {name}: {path}")
    if missing:
        print("[ExportConfig] ERROR: Checkpoint belum ada!")
        print("  Jalankan Training/CNNsaBiLSTM/Train.ipynb dan/atau Training/MViTv2/TrainMViT.ipynb dulu.\n")
        for m in missing:
            print(m)
        return False
    print("[ExportConfig] Semua checkpoint ditemukan.")
    return True


if __name__ == "__main__":
    create_output_dirs()
    verify_checkpoints_exist()
    print("CNNsa-BiLSTM checkpoint:", CNNSA_CHECKPOINT_PATH)
    print("MViTv2 checkpoint      :", MVIT_CHECKPOINT_PATH)
