"""
Exports the CNNsa-BiLSTM checkpoint
(Training/Output/cnnsa_bilstm/checkpoints/cnnsa_bilstm_best.pt) to
Export/Output/onnx/cnnsa_bilstm.onnx.

The model is rebuilt from the model_config stored in the checkpoint, so the
export matches the trained weights even if the config file changes later.
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
        print(f"[ExportCNNsa] ERROR: checkpoint not found: {CNNSA_CHECKPOINT_PATH}")
        print("  Run Training/CNNsaBiLSTM/Train.ipynb first.")
        return None

    print(f"[ExportCNNsa] Loading checkpoint: {CNNSA_CHECKPOINT_PATH}")
    checkpoint = torch.load(CNNSA_CHECKPOINT_PATH, map_location="cpu")
    model_config = checkpoint["model_config"]
    print(f"[ExportCNNsa] model_config from checkpoint: {model_config}")
    print(f"[ExportCNNsa] Validation accuracy of this checkpoint: {checkpoint.get('val_acc', 0)*100:.2f}%")

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
    model.eval()  # BatchNorm and Dropout must be in inference mode for export

    dummy_input = torch.randn(*CNNSA_INPUT_SHAPE)

    print(f"[ExportCNNsa] Exporting to ONNX (opset {CNNSA_OPSET}, input shape {CNNSA_INPUT_SHAPE})...")
    torch.onnx.export(
        model,
        dummy_input,
        str(CNNSA_ONNX_PATH),
        input_names=["keypoints"],
        output_names=["logits", "attn_weights"],
        opset_version=CNNSA_OPSET,
        dynamo=CNNSA_USE_DYNAMO,  # legacy TorchScript exporter
    )

    size_mb = CNNSA_ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"[ExportCNNsa] Done: {CNNSA_ONNX_PATH} ({size_mb:.2f} MB)")
    return CNNSA_ONNX_PATH


if __name__ == "__main__":
    export_cnnsa_bilstm()
