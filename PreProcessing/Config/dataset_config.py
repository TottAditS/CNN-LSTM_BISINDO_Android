"""
Dataset and preprocessing configuration.

All paths are resolved relative to the repository root, so the scripts run
from any working directory.
"""

from pathlib import Path

# This file lives at <ROOT>/PreProcessing/Config/dataset_config.py
_THIS_FILE = Path(__file__).resolve()
ROOT = _THIS_FILE.parent.parent.parent


def _raw(dataset_name: str) -> str:
    """Folder that holds the raw videos of a dataset."""
    return str(ROOT / "Datasets" / dataset_name)


def _processed(dataset_name: str) -> str:
    """Folder that receives the preprocessing output of a dataset."""
    return str(ROOT / "Datasets" / f"{dataset_name}_Processed")


# To add a dataset, copy the WLBisindo entry and edit its values.
DATASET_CONFIGS = {
    "WLBisindo": {
        "raw_video_dir" : _raw("WLBisindo"),
        "output_dir"    : _processed("WLBisindo"),
        "file_pattern"  : "signer{s}_label{l}_sample{n}.mp4",
        "num_signers"   : 5,       # signer0-signer4 (signer0 recorded 12 of the 32 labels)
        "num_labels"    : 32,      # label0-label31
        "num_samples"   : None,    # varies: 10 or 15 samples per (signer, label)
        "total_videos"  : 1600,
        "label_type"    : "numeric",
        "label_map"     : None,    # None = label_0, label_1, ...; or {0: "name", ...}
        "description"   : "Word-level BISINDO, 32 words, 5 signers, 10-15 samples",
    },
}


PREPROCESSING_CONFIG = {
    "frame": {
        "target_fps"  : 15,           # FPS after uniform sampling
        "resize"      : (224, 224),   # (H, W)
        "blur_kernel" : (5, 5),       # Gaussian blur kernel
        "blur_sigma"  : 0,            # 0 = derived from the kernel size
        "max_frames"  : 60,           # frames kept per video
        "min_frames"  : 10,           # shorter videos are skipped
    },
    # Splitting and augmentation happen later, on keypoints, in FeatureExtraction.
    "dataloader": {
        "save_frames_as_images": False,  # True also writes one .jpg per frame (debugging)
    },
}


if __name__ == "__main__":
    print(f"ROOT        : {ROOT}")
    for name, cfg in DATASET_CONFIGS.items():
        print(f"\n[{name}]")
        print(f"  Raw dir  : {cfg['raw_video_dir']}")
        print(f"  Output   : {cfg['output_dir']}")
