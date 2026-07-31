"""
Training/Config/shared_config.py
=================================
Konfigurasi yang dipakai BERSAMA oleh semua model di Training/ (CNNsaBiLSTM/, MViTv2/, dst)
dan oleh Training/Scripts/comparator.py.

Cuma berisi lokasi output "results" -- tempat tiap model menyimpan *_evaluation_summary.json
setelah eval, yang lalu dibaca comparator.py untuk bikin tabel perbandingan. Config yang
spesifik ke satu model (hyperparameter, path checkpoint/log model itu sendiri) tetap hidup
di Config/ masing-masing model (mis. Training/CNNsaBiLSTM/Config/train_config.py).
"""

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

TRAINING_OUTPUT_DIR = ROOT / "Training" / "Output"
RESULTS_DIR = TRAINING_OUTPUT_DIR / "results"


def create_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
