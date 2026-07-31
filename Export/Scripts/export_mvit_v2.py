"""
Export/Scripts/export_mvit_v2.py
===================================
Export checkpoint MViTv2 (Training/Output/mvit_v2/checkpoints/mvit_v2_best.pt) ke ONNX
(Export/Output/onnx/mvit_v2.onnx).

Catatan: model dibangun dengan pretrained=False -- bobot sesungguhnya datang dari checkpoint
hasil fine-tune, bukan Kinetics-400 lagi (tidak perlu network/download saat export).

PENTING -- pakai exporter BARU (dynamo=True), bukan legacy: exporter legacy (dipakai untuk
CNNsa-BiLSTM) menghasilkan model yang "sukses" di-export tapi GAGAL dijalankan di ONNX Runtime
("Loop subgraph input 0 has unknown shape") -- pooling-attention MViT ke-trace jadi Loop node
dinamis yang shape-nya tidak bisa di-infer exporter lama. Sudah dikonfirmasi lewat tes langsung:
exporter baru berhasil (selisih vs PyTorch ~1e-6). Exporter baru butuh package `onnxscript`
(sudah ditambahkan ke Requirements.txt).
"""

import sys
from pathlib import Path

# Exporter baru (dynamo=True) mencetak progress pakai karakter unicode (mis. checkmark) --
# bisa UnicodeEncodeError di konsol Windows yang codepage-nya bukan UTF-8 (masalah yang sama
# yang sudah kita perbaiki di kode kita sendiri sebelumnya). reconfigure() di sini bikin script
# ini aman jalan dari console manapun tanpa perlu PYTHONIOENCODING manual.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Export.Config.export_config import (
    MVIT_CHECKPOINT_PATH, MVIT_ONNX_PATH, MVIT_INPUT_SHAPE,
    MVIT_OPSET, MVIT_USE_DYNAMO, create_output_dirs,
)
from Training.MViTv2.Models.mvit_v2 import build_mvit_v2


def export_mvit_v2():
    create_output_dirs()

    if not MVIT_CHECKPOINT_PATH.exists():
        print(f"[ExportMViT] ERROR: Checkpoint tidak ditemukan: {MVIT_CHECKPOINT_PATH}")
        print("  Jalankan Training/MViTv2/TrainMViT.ipynb terlebih dahulu.")
        return None

    print(f"[ExportMViT] Loading checkpoint: {MVIT_CHECKPOINT_PATH}")
    checkpoint = torch.load(MVIT_CHECKPOINT_PATH, map_location="cpu")
    model_config = checkpoint["model_config"]
    print(f"[ExportMViT] model_config dari checkpoint: {model_config}")
    print(f"[ExportMViT] Val Acc checkpoint ini saat training: {checkpoint.get('val_acc', 0)*100:.2f}%")

    model = build_mvit_v2(
        num_classes=model_config["num_classes"],
        freeze_until_block=model_config["freeze_until_block"],
        pretrained=False,  # bobot dari checkpoint fine-tune, bukan Kinetics-400 lagi
        dropout=model_config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(*MVIT_INPUT_SHAPE)

    print(f"[ExportMViT] Export ke ONNX (opset {MVIT_OPSET}, dynamo={MVIT_USE_DYNAMO}, "
          f"shape input {MVIT_INPUT_SHAPE})...")
    print("  (model besar, proses tracing bisa makan waktu beberapa menit)")
    torch.onnx.export(
        model,
        dummy_input,
        str(MVIT_ONNX_PATH),
        input_names=["video_clip"],
        output_names=["logits"],
        opset_version=MVIT_OPSET,
        dynamo=MVIT_USE_DYNAMO,  # exporter baru -- WAJIB untuk model ini, lihat docstring di atas
    )

    size_mb = MVIT_ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"[ExportMViT] Selesai! {MVIT_ONNX_PATH} ({size_mb:.2f} MB)")
    return MVIT_ONNX_PATH


if __name__ == "__main__":
    export_mvit_v2()
