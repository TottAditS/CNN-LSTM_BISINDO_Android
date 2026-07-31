"""
Dataset Configuration
======================
Path otomatis dihitung relatif dari ROOT repo (CNN-LSTM_BISINDO_Android/).
Tidak perlu hardcode path absolut -- script jalan di mana saja.

Struktur repo yang diasumsikan:
    CNN-LSTM_BISINDO_Android/
    +-- Datasets/
    |   +-- WLBisindo/          <- raw video
    +-- PreProcessing/
    |   +-- Config/
    |   |   +-- dataset_config.py   <- FILE INI
    |   +-- Scripts/
    |   +-- PreProcess.ipynb
    +-- Requirements.txt
"""

from pathlib import Path

# -- Auto-detect ROOT repo --------------------------------------------------
# File ini ada di: <ROOT>/PreProcessing/Config/dataset_config.py
# Jadi ROOT = 2 level ke atas dari file ini
_THIS_FILE = Path(__file__).resolve()
ROOT = _THIS_FILE.parent.parent.parent   # Config/ -> PreProcessing/ -> ROOT

# -- Helper ----------------------------------------------------------------
def _raw(dataset_name: str) -> str:
    """Path ke folder raw video dataset."""
    return str(ROOT / "Datasets" / dataset_name)

def _processed(dataset_name: str) -> str:
    """Path ke folder output hasil preprocessing."""
    return str(ROOT / "Datasets" / f"{dataset_name}_Processed")


# ==========================================================================
# DATASET CONFIGS
# Untuk menambah dataset baru: copy-paste blok WLBisindo dan edit nilainya
# ==========================================================================
DATASET_CONFIGS = {

    # -- Dataset 1: WLBisindo ----------------------------------------------
    "WLBisindo": {
        "raw_video_dir" : _raw("WLBisindo"),
        "output_dir"    : _processed("WLBisindo"),
        "file_pattern"  : "signer{s}_label{l}_sample{n}.mp4",
        "num_signers"   : 5,       # signer0-signer4 (signer0 baru punya 12/32 label)
        "num_labels"    : 32,      # label0-label31 yang benar-benar ada di dataset saat ini
        "num_samples"   : None,    # bervariasi: 10 atau 15 sample per (signer, label)
        "total_videos"  : 1600,
        "label_type"    : "numeric",
        "label_map"     : None,   # None = auto (label_0, label_1, ...)
                                  # Atau: {0: "halo", 1: "terima kasih", ...}
        "description"   : "BISINDO Word-Level, 32 kata, 5 signer, 10-15 sample",
    },

    # -- Dataset 2 (contoh, uncomment jika ada) ----------------------------
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


# ==========================================================================
# GLOBAL PREPROCESSING SETTINGS
# ==========================================================================
PREPROCESSING_CONFIG = {

    "frame": {
        "target_fps"  : 15,           # FPS setelah uniform sampling
        "resize"      : (224, 224),   # (H, W) target ukuran frame
        "blur_kernel" : (5, 5),       # Gaussian blur kernel
        "blur_sigma"  : 0,            # 0 = auto
        "max_frames"  : 60,           # Maks frame per video
        "min_frames"  : 10,           # Video < min_frames di-skip
    },

    # Catatan: augmentasi & split train/val/test TIDAK lagi dilakukan di sini.
    # PreProcessing hanya extract frame mentah. Augmentasi (flip/rotate/zoom)
    # dan split dataset sekarang dilakukan di FeatureExtraction pada level
    # keypoint (lihat FeatureExtraction/Config/feature_config.py -> AUGMENTATION_CONFIG),
    # karena model final memakai keypoint MediaPipe, bukan raw pixel frame.
    "dataloader": {
        "save_frames_as_images": False,  # True = simpan .jpg per frame (debug)
    },
}


# -- Debug: print paths saat diimport --------------------------------------
if __name__ == "__main__":
    print(f"ROOT repo   : {ROOT}")
    for name, cfg in DATASET_CONFIGS.items():
        print(f"\n[{name}]")
        print(f"  Raw dir  : {cfg['raw_video_dir']}")
        print(f"  Output   : {cfg['output_dir']}")