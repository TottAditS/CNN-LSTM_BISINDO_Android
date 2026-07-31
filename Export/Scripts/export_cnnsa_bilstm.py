"""
Export/Scripts/export_cnnsa_bilstm.py
========================================
Export checkpoint CNNsa-BiLSTM (Training/Output/cnnsa_bilstm/checkpoints/cnnsa_bilstm_best.pt)
ke ONNX (Export/Output/onnx/cnnsa_bilstm.onnx).

Arsitektur model dibangun ulang dari "model_config" yang tersimpan DI DALAM checkpoint itu
sendiri (bukan re-import dari Training/CNNsaBiLSTM/Config/train_config.py) -- supaya export
selalu cocok persis dengan model yang benar-benar dilatih, walaupun config berubah setelahnya.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Export.Config.export_config import (
    CNNSA_CHECKPOINT_PATH, CNNSA_ONNX_PATH, CNNSA_INPUT_SHAPE,
    CNNSA_OPSET, CNNSA_USE_DYNAMO, create_output_dirs,
)
from Training.CNNsaBiLSTM.Models.cnnsa_bilstm import CNNsaBiLSTM


def export_cnnsa_bilstm():
    create_output_dirs()

    if not CNNSA_CHECKPOINT_PATH.exists():
        print(f"[ExportCNNsa] ERROR: Checkpoint tidak ditemukan: {CNNSA_CHECKPOINT_PATH}")
        print("  Jalankan Training/CNNsaBiLSTM/Train.ipynb terlebih dahulu.")
        return None

    print(f"[ExportCNNsa] Loading checkpoint: {CNNSA_CHECKPOINT_PATH}")
    checkpoint = torch.load(CNNSA_CHECKPOINT_PATH, map_location="cpu")
    model_config = checkpoint["model_config"]
    print(f"[ExportCNNsa] model_config dari checkpoint: {model_config}")
    print(f"[ExportCNNsa] Val Acc checkpoint ini saat training: {checkpoint.get('val_acc', 0)*100:.2f}%")

    model = CNNsaBiLSTM(
        input_size=model_config["input_size"],
        feature_dim=model_config["feature_dim"],
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        num_classes=model_config["num_classes"],
        dropout=model_config["dropout"],
        attn_reduction=model_config["attn_reduction"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # WAJIB eval() -- BatchNorm/Dropout harus mode inferensi saat export

    dummy_input = torch.randn(*CNNSA_INPUT_SHAPE)

    print(f"[ExportCNNsa] Export ke ONNX (opset {CNNSA_OPSET}, shape input {CNNSA_INPUT_SHAPE})...")
    torch.onnx.export(
        model,
        dummy_input,
        str(CNNSA_ONNX_PATH),
        input_names=["keypoints"],
        output_names=["logits", "attn_weights"],
        opset_version=CNNSA_OPSET,
        dynamo=CNNSA_USE_DYNAMO,  # legacy TorchScript-based exporter -- dikonfirmasi jalan untuk model ini
    )

    size_mb = CNNSA_ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"[ExportCNNsa] Selesai! {CNNSA_ONNX_PATH} ({size_mb:.2f} MB)")
    return CNNSA_ONNX_PATH


if __name__ == "__main__":
    export_cnnsa_bilstm()
