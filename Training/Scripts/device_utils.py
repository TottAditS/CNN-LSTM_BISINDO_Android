"""
Device selection shared by both trainers.
"""

import torch


def get_device(verbose: bool = True) -> torch.device:
    """
    Return the CUDA device if available, otherwise the CPU with a clear warning,
    so a CPU-only PyTorch install is noticed before a long training run.
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        if verbose:
            print(f"[Device] GPU detected  : {name} (compute capability sm_{major}{minor})")
            print(f"[Device] CUDA runtime  : {torch.version.cuda} (PyTorch build)")
        return torch.device("cuda")

    if verbose:
        print("=" * 64)
        print("[Device] CUDA not available, training will run on the CPU (much slower).")
        print("  Possible causes:")
        print("  1. A CPU-only PyTorch build is installed (not +cu128).")
        print("  2. The CUDA wheel does not support this GPU's compute capability")
        print("     (RTX 50-series GPUs need the cu128 wheel).")
        print("  3. The NVIDIA driver is missing or outdated.")
        print('  Check: python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"')
        print("  See Requirements.txt for the CUDA wheel index.")
        print("=" * 64)
    return torch.device("cpu")
