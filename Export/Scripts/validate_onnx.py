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

Selain print ke console, script ini juga MENYIMPAN hasilnya:
- Tabular : Export/Output/validate_onnx_summary.json
- Visual  : Export/Output/charts/onnx_latency_comparison.png (bar chart PyTorch vs ONNX
            Runtime, kedua model)
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Export.Config.export_config import (
    CNNSA_ONNX_PATH, MVIT_ONNX_PATH, CNNSA_INPUT_SHAPE, MVIT_INPUT_SHAPE,
    EXPORT_OUTPUT_DIR, EXPORT_CHARTS_DIR, create_output_dirs,
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


def _compare(name: str, pytorch_out: np.ndarray, onnx_out: np.ndarray) -> tuple[bool, float, bool]:
    max_diff = float(np.abs(pytorch_out - onnx_out).max())
    same_pred = int(np.argmax(pytorch_out)) == int(np.argmax(onnx_out))
    ok = max_diff < ATOL and same_pred

    print(f"\n[ValidateONNX] === {name} ===")
    print(f"  Max abs diff (logits) : {max_diff:.8f} (threshold: {ATOL})")
    print(f"  Prediksi kelas sama?  : {same_pred} "
          f"(PyTorch={int(np.argmax(pytorch_out))}, ONNX={int(np.argmax(onnx_out))})")
    print(f"  STATUS                : {'PASS' if ok else 'FAIL'}")
    return ok, max_diff, same_pred


def _benchmark(fn, num_runs: int = 20) -> float:
    for _ in range(3):
        fn()
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def validate_cnnsa_bilstm() -> dict:
    if not CNNSA_ONNX_PATH.exists():
        print(f"[ValidateONNX] ERROR: {CNNSA_ONNX_PATH} tidak ditemukan. Jalankan export_cnnsa_bilstm.py dulu.")
        return {"ok": False}

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

    ok, max_diff, same_pred = _compare("CNNsa-BiLSTM", pytorch_logits, onnx_logits)

    pt_ms = _benchmark(lambda: model(x))
    onnx_ms = _benchmark(lambda: sess.run(None, {"keypoints": x_np}))
    print(f"  Latensi PyTorch (CPU) : {pt_ms:.3f} ms")
    print(f"  Latensi ONNX RT (CPU) : {onnx_ms:.3f} ms")

    return {
        "ok": ok, "max_abs_diff": max_diff, "same_prediction": same_pred,
        "pytorch_ms": pt_ms, "onnx_ms": onnx_ms, "speedup": pt_ms / onnx_ms,
    }


def validate_mvit_v2() -> dict:
    if not MVIT_ONNX_PATH.exists():
        print(f"[ValidateONNX] ERROR: {MVIT_ONNX_PATH} tidak ditemukan. Jalankan export_mvit_v2.py dulu.")
        return {"ok": False}

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

    ok, max_diff, same_pred = _compare("MViTv2", pytorch_logits, onnx_logits)

    pt_ms = _benchmark(lambda: model(x), num_runs=5)
    onnx_ms = _benchmark(lambda: sess.run(None, {"video_clip": x_np}), num_runs=5)
    print(f"  Latensi PyTorch (CPU) : {pt_ms:.3f} ms")
    print(f"  Latensi ONNX RT (CPU) : {onnx_ms:.3f} ms")

    return {
        "ok": ok, "max_abs_diff": max_diff, "same_prediction": same_pred,
        "pytorch_ms": pt_ms, "onnx_ms": onnx_ms, "speedup": pt_ms / onnx_ms,
    }


def save_validation_summary(cnnsa_result: dict, mvit_result: dict) -> Path:
    out_path = EXPORT_OUTPUT_DIR / "validate_onnx_summary.json"
    with open(out_path, "w") as f:
        json.dump({"cnnsa_bilstm": cnnsa_result, "mvit_v2": mvit_result}, f, indent=2)
    print(f"\n[ValidateONNX] Ringkasan validasi disimpan ke: {out_path}")
    return out_path


def plot_latency_comparison(cnnsa_result: dict, mvit_result: dict) -> Path | None:
    if not cnnsa_result.get("ok") or not mvit_result.get("ok"):
        print("[ValidateONNX] Salah satu validasi gagal/tidak lengkap, skip plot_latency_comparison.")
        return None

    import matplotlib.pyplot as plt
    import numpy as np

    models = ["CNNsa-BiLSTM", "MViTv2"]
    pt_vals = [cnnsa_result["pytorch_ms"], mvit_result["pytorch_ms"]]
    onnx_vals = [cnnsa_result["onnx_ms"], mvit_result["onnx_ms"]]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width / 2, pt_vals, width, label="PyTorch (CPU)", color="#EE4C2C")
    bars2 = ax.bar(x + width / 2, onnx_vals, width, label="ONNX Runtime (CPU)", color="#5C9BD1")

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f} ms", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_ylabel("Latensi (ms)")
    ax.set_yscale("log")
    ax.set_title("Latensi Inferensi CPU -- PyTorch vs ONNX Runtime")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    fig.tight_layout()

    out_path = EXPORT_CHARTS_DIR / "onnx_latency_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[ValidateONNX] Chart latensi disimpan ke: {out_path}")
    return out_path


def validate_all():
    create_output_dirs()
    print("[ValidateONNX] providers ONNX Runtime tersedia:", ort.get_available_providers())
    cnnsa_result = validate_cnnsa_bilstm()
    mvit_result = validate_mvit_v2()
    print("\n[ValidateONNX] === Ringkasan ===")
    print(f"  CNNsa-BiLSTM: {'PASS' if cnnsa_result.get('ok') else 'FAIL'}")
    print(f"  MViTv2      : {'PASS' if mvit_result.get('ok') else 'FAIL'}")

    save_validation_summary(cnnsa_result, mvit_result)
    plot_latency_comparison(cnnsa_result, mvit_result)

    return cnnsa_result.get("ok", False) and mvit_result.get("ok", False)


if __name__ == "__main__":
    validate_all()
