"""
Exports the MViTv2 checkpoint (Training/Output/mvit_v2/checkpoints/mvit_v2_best.pt)
to Export/Output/onnx/mvit_v2.onnx plus its external weight file mvit_v2.onnx.data.

MViTv2 needs the dynamo-based exporter (dynamo=True, requires onnxscript). The
legacy exporter produces a file that fails in ONNX Runtime with
"Loop subgraph input 0 has unknown shape", caused by the pooling attention.
"""

import sys
from pathlib import Path

# The dynamo exporter prints non-ASCII progress symbols, which raise UnicodeEncodeError
# on Windows consoles that do not use UTF-8.
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
        print(f"[ExportMViT] ERROR: checkpoint not found: {MVIT_CHECKPOINT_PATH}")
        print("  Run Training/MViTv2/TrainMViT.ipynb first.")
        return None

    print(f"[ExportMViT] Loading checkpoint: {MVIT_CHECKPOINT_PATH}")
    checkpoint = torch.load(MVIT_CHECKPOINT_PATH, map_location="cpu")
    model_config = checkpoint["model_config"]
    print(f"[ExportMViT] model_config from checkpoint: {model_config}")
    print(f"[ExportMViT] Validation accuracy of this checkpoint: {checkpoint.get('val_acc', 0)*100:.2f}%")

    model = build_mvit_v2(
        num_classes=model_config["num_classes"],
        freeze_until_block=model_config["freeze_until_block"],
        pretrained=False,  # weights come from the fine-tuned checkpoint
        dropout=model_config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(*MVIT_INPUT_SHAPE)

    print(f"[ExportMViT] Exporting to ONNX (opset {MVIT_OPSET}, dynamo={MVIT_USE_DYNAMO}, "
          f"input shape {MVIT_INPUT_SHAPE})...")
    print("  (large model, tracing can take a few minutes)")
    torch.onnx.export(
        model,
        dummy_input,
        str(MVIT_ONNX_PATH),
        input_names=["video_clip"],
        output_names=["logits"],
        opset_version=MVIT_OPSET,
        dynamo=MVIT_USE_DYNAMO,  # required for this model, see the module docstring
    )

    size_mb = MVIT_ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"[ExportMViT] Done: {MVIT_ONNX_PATH} ({size_mb:.2f} MB)")
    return MVIT_ONNX_PATH


if __name__ == "__main__":
    export_mvit_v2()
