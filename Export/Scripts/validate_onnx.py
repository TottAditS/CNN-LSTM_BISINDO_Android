"""
Export/Scripts/validate_onnx.py
==================================
Validasi hasil export ONNX: bandingkan output model PyTorch asli vs model ONNX (dijalankan
lewat ONNX Runtime, CPUExecutionProvider) memakai input YANG SAMA.

CPU dipilih sengaja (bukan GPU): (1) ini yang benar-benar dites & jalan di mesin ini --
`onnxruntime-gpu` butuh CUDA 13 + cuDNN 9 yang belum terpasang di sistem, beda dari PyTorch
yang bundle CUDA 12.8 sendiri di dalam wheel-nya; (2) realistisnya Android juga default jalan
CPU (atau NNAPI/GPU delegate khusus, bukan CUDA ala desktop), jadi validasi CPU lebih relevan
ke kondisi target deployment.

Kalau ada data test asli (frame_manifest.json / test_sequences.npz), dipakai satu sample
sungguhan -- kalau belum ada (mis. baru mau coba export tanpa jalanin training/data dulu),
fallback ke random dummy input.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Export.Config.export_config import (
    CNNSA_ONNX_PATH, MVIT_ONNX_PATH, CNNSA_INPUT_SHAPE, MVIT_INPUT_SHAPE,
)

ATOL = 1e-4  # toleransi selisih absolut PyTorch vs ONNX Runtime (float32, beda backend wajar ada sedikit selisih)


def _get_cnnsa_sample():
    """Ambil satu sample asli dari test_sequences.npz kalau ada, else random."""
    try:
        from Training.CNNsaBiLSTM.Config.train_config import TEST_NPZ
        if TEST_NPZ.exists():
            data = np.load(TEST_NPZ, allow_pickle=True)
            x = data["X"][0:1].astype(np.float32)  # (1, 30, 126)
            print(f"[ValidateONNX] CNNsa-BiLSTM: pakai sample asli dari {TEST_NPZ.name} (label={data['y'][0]})")
            return torch.from_numpy(x)
    except Exception as e:
        print(f"[ValidateONNX] CNNsa-BiLSTM: gagal load sample asli ({e}), pakai random dummy.")
    return torch.randn(*CNNSA_INPUT_SHAPE)


def _get_mvit_sample():
    """Ambil satu clip video asli dari test split MViTv2 kalau ada, else random."""
    try:
        from Training.MViTv2.Scripts.mvit_dataset import build_mvit_datasets
        _, _, test_ds = build_mvit_datasets()
        if len(test_ds) > 0:
            clip, label = test_ds[0]
            print(f"[ValidateONNX] MViTv2: pakai sample asli dari test split (label={label})")
            return clip.unsqueeze(0)  # (1, 3, 16, 224, 224)
    except Exception as e:
        print(f"[ValidateONNX] MViTv2: gagal load sample asli ({e}), pakai random dummy.")
    return torch.randn(*MVIT_INPUT_SHAPE)


def _compare(name: str, pytorch_out: np.ndarray, onnx_out: np.ndarray) -> bool:
    max_diff = float(np.abs(pytorch_out - onnx_out).max())
    same_pred = int(np.argmax(pytorch_out)) == int(np.argmax(onnx_out))
    ok = max_diff < ATOL and same_pred

    print(f"\n[ValidateONNX] === {name} ===")
    print(f"  Max abs diff (logits) : {max_diff:.8f} (threshold: {ATOL})")
    print(f"  Prediksi kelas sama?  : {same_pred} "
          f"(PyTorch={int(np.argmax(pytorch_out))}, ONNX={int(np.argmax(onnx_out))})")
    print(f"  STATUS                : {'PASS' if ok else 'FAIL'}")
    return ok


def _benchmark(fn, num_runs: int = 20) -> float:
    for _ in range(3):
        fn()
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def validate_cnnsa_bilstm() -> bool:
    if not CNNSA_ONNX_PATH.exists():
        print(f"[ValidateONNX] ERROR: {CNNSA_ONNX_PATH} tidak ditemukan. Jalankan export_cnnsa_bilstm.py dulu.")
        return False

    from Training.CNNsaBiLSTM.Config.train_config import CHECKPOINT_DIR, CHECKPOINT_CONFIG
    from Training.CNNsaBiLSTM.Models.cnnsa_bilstm import CNNsaBiLSTM

    checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_CONFIG["filename"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint["model_config"]

    model = CNNsaBiLSTM(
        input_size=model_config["input_size"], feature_dim=model_config["feature_dim"],
        hidden_size=model_config["hidden_size"], num_layers=model_config["num_layers"],
        num_classes=model_config["num_classes"], dropout=model_config["dropout"],
        attn_reduction=model_config["attn_reduction"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = _get_cnnsa_sample()
    x_np = x.numpy()

    with torch.no_grad():
        pytorch_logits, _ = model(x)
    pytorch_logits = pytorch_logits.numpy()

    sess = ort.InferenceSession(str(CNNSA_ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_logits, _ = sess.run(None, {"keypoints": x_np})

    ok = _compare("CNNsa-BiLSTM", pytorch_logits, onnx_logits)

    pt_ms = _benchmark(lambda: model(x))
    onnx_ms = _benchmark(lambda: sess.run(None, {"keypoints": x_np}))
    print(f"  Latensi PyTorch (CPU) : {pt_ms:.3f} ms")
    print(f"  Latensi ONNX RT (CPU) : {onnx_ms:.3f} ms")

    return ok


def validate_mvit_v2() -> bool:
    if not MVIT_ONNX_PATH.exists():
        print(f"[ValidateONNX] ERROR: {MVIT_ONNX_PATH} tidak ditemukan. Jalankan export_mvit_v2.py dulu.")
        return False

    from Training.MViTv2.Config.mvit_config import CHECKPOINT_DIR, CHECKPOINT_CONFIG
    from Training.MViTv2.Models.mvit_v2 import build_mvit_v2

    checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_CONFIG["filename"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint["model_config"]

    model = build_mvit_v2(
        num_classes=model_config["num_classes"], freeze_until_block=model_config["freeze_until_block"],
        pretrained=False, dropout=model_config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = _get_mvit_sample()
    x_np = x.numpy()

    with torch.no_grad():
        pytorch_logits = model(x)
    pytorch_logits = pytorch_logits.numpy()

    sess = ort.InferenceSession(str(MVIT_ONNX_PATH), providers=["CPUExecutionProvider"])
    (onnx_logits,) = sess.run(None, {"video_clip": x_np})

    ok = _compare("MViTv2", pytorch_logits, onnx_logits)

    pt_ms = _benchmark(lambda: model(x), num_runs=5)
    onnx_ms = _benchmark(lambda: sess.run(None, {"video_clip": x_np}), num_runs=5)
    print(f"  Latensi PyTorch (CPU) : {pt_ms:.3f} ms")
    print(f"  Latensi ONNX RT (CPU) : {onnx_ms:.3f} ms")

    return ok


def validate_all():
    print("[ValidateONNX] providers ONNX Runtime tersedia:", ort.get_available_providers())
    ok1 = validate_cnnsa_bilstm()
    ok2 = validate_mvit_v2()
    print("\n[ValidateONNX] === Ringkasan ===")
    print(f"  CNNsa-BiLSTM: {'PASS' if ok1 else 'FAIL'}")
    print(f"  MViTv2      : {'PASS' if ok2 else 'FAIL'}")
    return ok1 and ok2


if __name__ == "__main__":
    validate_all()
