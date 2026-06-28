"""
Dataset Configuration
======================
Path otomatis dihitung relatif dari ROOT repo (CNN-LSTM_BISINDO_Android/).
Tidak perlu hardcode path absolut — script jalan di mana saja.

Struktur repo yang diasumsikan:
    CNN-LSTM_BISINDO_Android/
    ├── Datasets/
    │   └── WLBisindo/          ← raw video
    ├── PreProcessing/
    │   ├── Config/
    │   │   └── dataset_config.py   ← FILE INI
    │   ├── Scripts/
    │   └── PreProcess.ipynb
    └── Requirements.txt
"""

from pathlib import Path

# ── Auto-detect ROOT repo ──────────────────────────────────────────────────
# File ini ada di: <ROOT>/PreProcessing/Config/dataset_config.py
# Jadi ROOT = 2 level ke atas dari file ini
_THIS_FILE = Path(__file__).resolve()
ROOT = _THIS_FILE.parent.parent.parent   # Config/ → PreProcessing/ → ROOT

# ── Helper ────────────────────────────────────────────────────────────────
def _raw(dataset_name: str) -> str:
    """Path ke folder raw video dataset."""
    return str(ROOT / "Datasets" / dataset_name)

def _processed(dataset_name: str) -> str:
    """Path ke folder output hasil preprocessing."""
    return str(ROOT / "Datasets" / f"{dataset_name}_Processed")


# ══════════════════════════════════════════════════════════════════════════
# DATASET CONFIGS
# Untuk menambah dataset baru: copy-paste blok WLBisindo dan edit nilainya
# ══════════════════════════════════════════════════════════════════════════
DATASET_CONFIGS = {

    # ── Dataset 1: WLBisindo ──────────────────────────────────────────────
    "WLBisindo": {
        "raw_video_dir" : _raw("WLBisindo"),
        "output_dir"    : _processed("WLBisindo"),
        "file_pattern"  : "signer{s}_label{l}_sample{n}.mp4",
        "num_signers"   : 4,
        "num_labels"    : 40,
        "num_samples"   : 10,
        "total_videos"  : 1600,
        "label_type"    : "numeric",
        "label_map"     : None,   # None = auto (label_0, label_1, ...)
                                  # Atau: {0: "halo", 1: "terima kasih", ...}
        "fps_target"    : 30,
        "description"   : "BISINDO Word-Level, 40 kata, 4 signer, 10 sample",
    },

    # ── Dataset 2 (contoh, uncomment jika ada) ────────────────────────────
    # "SIBI_Alphabet": {
    #     "raw_video_dir" : _raw("SIBI_Alphabet"),
    #     "output_dir"    : _processed("SIBI_Alphabet"),
    #     "file_pattern"  : "{label}_{signer}_{sample}.mp4",
    #     "num_signers"   : 5,
    #     "num_labels"    : 26,
    #     "num_samples"   : 20,
    #     "total_videos"  : 2600,
    #     "label_type"    : "string",
    #     "label_map"     : None,
    #     "fps_target"    : 30,
    #     "description"   : "SIBI Alphabet, 26 huruf",
    # },
}


# ══════════════════════════════════════════════════════════════════════════
# GLOBAL PREPROCESSING SETTINGS
# ══════════════════════════════════════════════════════════════════════════
PREPROCESSING_CONFIG = {

    "frame": {
        "target_fps"  : 15,           # FPS setelah uniform sampling
        "resize"      : (224, 224),   # (H, W) target ukuran frame
        "blur_kernel" : (5, 5),       # Gaussian blur kernel
        "blur_sigma"  : 0,            # 0 = auto
        "max_frames"  : 60,           # Maks frame per video
        "min_frames"  : 10,           # Video < min_frames di-skip
    },

    "augmentation": {
        "enabled"            : True,
        "techniques"         : [
            "horizontal_flip",
            "brightness",
            "contrast",
            "rotation",
            "zoom",
        ],
        "horizontal_flip_prob" : 0.5,
        "brightness_range"     : (0.7, 1.3),
        "contrast_range"       : (0.8, 1.2),
        "rotation_range"       : (-10, 10),
        "zoom_range"           : (0.9, 1.1),
        "augment_multiplier"   : 2,   # 1 original + 2 augmented = 3x data
    },

    "sequence": {
        "fixed_length"  : 30,          # Semua sequence → N frame
        "padding_mode"  : "zero",      # 'zero' atau 'repeat_last'
        "truncate_mode" : "center",    # 'start', 'end', 'center'
    },

    "dataloader": {
        "train_ratio"         : 0.7,
        "val_ratio"           : 0.15,
        "test_ratio"          : 0.15,
        "shuffle"             : True,
        "random_seed"         : 42,
        "save_format"         : "npz",
        "save_frames_as_images": False,  # True = simpan .jpg per frame (debug)
    },
}


# ── Debug: print paths saat diimport ──────────────────────────────────────
if __name__ == "__main__":
    print(f"ROOT repo   : {ROOT}")
    for name, cfg in DATASET_CONFIGS.items():
        print(f"\n[{name}]")
        print(f"  Raw dir  : {cfg['raw_video_dir']}")
        print(f"  Output   : {cfg['output_dir']}")