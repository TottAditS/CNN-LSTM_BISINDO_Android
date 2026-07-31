"""
Training/Scripts/device_utils.py
==================================
Utilitas device (CPU/GPU) yang dipakai bersama oleh semua trainer di Training/
(CNNsaBiLSTM/Scripts/trainer.py, MViTv2/Scripts/mvit_trainer.py, dst).
"""

import torch


def get_device(verbose: bool = True) -> torch.device:
    """
    Pilih GPU (CUDA) jika tersedia, fallback ke CPU dengan warning eksplisit.
    Dipanggil di awal training supaya silent CPU fallback (mis. karena wheel
    PyTorch yang terinstall tidak cocok dengan compute capability GPU) langsung
    ketahuan, bukan baru sadar setelah training berjam-jam di CPU.
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        if verbose:
            print(f"[Device] GPU terdeteksi : {name} (compute capability sm_{major}{minor})")
            print(f"[Device] CUDA runtime   : {torch.version.cuda} (build PyTorch)")
        return torch.device("cuda")

    if verbose:
        print("=" * 64)
        print("[Device] CUDA TIDAK terdeteksi -- training akan jalan di CPU (jauh lebih lambat).")
        print("  Kemungkinan penyebab:")
        print("  1. PyTorch yang terinstall adalah build CPU-only (bukan +cu128).")
        print("  2. Wheel CUDA tidak cocok dengan compute capability GPU kamu")
        print("     (GPU Blackwell/RTX 50-series butuh wheel cu128; wheel cu121 lama TIDAK cukup).")
        print("  3. Driver NVIDIA belum terinstall / perlu update.")
        print('  Cek: python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"')
        print("  Lihat Requirements.txt untuk instruksi install wheel CUDA yang benar.")
        print("=" * 64)
    return torch.device("cpu")
