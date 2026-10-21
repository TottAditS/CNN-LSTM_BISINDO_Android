"""
MViTv2 dataset: loads the RGB frames from PreProcessing, samples or pads them
to CLIP_LENGTH, normalizes with the Kinetics-400 mean and std, and applies a
random horizontal flip to the training split.

The video-level split comes from FeatureExtraction/Scripts/video_split.py, the
same function used for CNNsa-BiLSTM, so both models share the val/test videos.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Training.MViTv2.Config.mvit_config import (
    FRAME_MANIFEST, CLIP_LENGTH, KINETICS_MEAN, KINETICS_STD,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
)
from FeatureExtraction.Scripts.video_split import stratified_video_split

MEAN = np.array(KINETICS_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)  # broadcasts over (T, H, W, C)
STD  = np.array(KINETICS_STD, dtype=np.float32).reshape(1, 1, 1, 3)


def _temporal_sample_or_pad(frames: np.ndarray, clip_length: int) -> np.ndarray:
    """
    Resize a video in time to clip_length frames.
    - T > clip_length: uniform sampling across the video
    - T < clip_length: repeat the last frame

    Args:
        frames: (T, H, W, C) uint8
    Returns:
        (clip_length, H, W, C) uint8
    """
    T = frames.shape[0]
    if T == clip_length:
        return frames
    elif T > clip_length:
        indices = np.linspace(0, T - 1, clip_length).round().astype(int)
        return frames[indices]
    else:
        pad_count = clip_length - T
        pad = np.repeat(frames[-1:], pad_count, axis=0)
        return np.concatenate([frames, pad], axis=0)


def _horizontal_flip(frames: np.ndarray) -> np.ndarray:
    """Mirror every frame horizontally. frames: (T, H, W, C)."""
    return frames[:, :, ::-1, :].copy()


class VideoClipDataset(Dataset):
    """
    Args:
        items    : frame_manifest.json items of one split
        is_train : enables the random horizontal flip
    """

    def __init__(self, items: List[Dict], is_train: bool = False, flip_prob: float = 0.5):
        self.items = items
        self.is_train = is_train
        self.flip_prob = flip_prob
        self.rng = np.random.default_rng(RANDOM_SEED)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.items[idx]
        frames = np.load(item["frames_npy"])  # (T, H, W, C) uint8 RGB

        frames = _temporal_sample_or_pad(frames, CLIP_LENGTH)

        if self.is_train and self.rng.random() < self.flip_prob:
            frames = _horizontal_flip(frames)

        # uint8 [0, 255] -> float32 [0, 1] -> Kinetics normalization
        clip = frames.astype(np.float32) / 255.0
        clip = (clip - MEAN) / STD

        # (T, H, W, C) -> (C, T, H, W), the layout of torchvision video models
        clip = np.transpose(clip, (3, 0, 1, 2))

        return torch.from_numpy(clip).float(), int(item["label_idx"])


def load_manifest() -> List[Dict]:
    if not FRAME_MANIFEST.exists():
        return []
    with open(FRAME_MANIFEST, "r") as f:
        return json.load(f)


def build_mvit_datasets() -> Tuple[VideoClipDataset, VideoClipDataset, VideoClipDataset]:
    """Load the manifest, apply the shared split, and build one dataset per split."""
    manifest = load_manifest()
    if not manifest:
        raise FileNotFoundError(
            f"frame_manifest.json not found at {FRAME_MANIFEST}. "
            "Run PreProcessing/PreProcess.ipynb first."
        )

    train_items, val_items, test_items = stratified_video_split(
        manifest, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    train_ds = VideoClipDataset(train_items, is_train=True)
    val_ds   = VideoClipDataset(val_items, is_train=False)
    test_ds  = VideoClipDataset(test_items, is_train=False)

    print(f"[MViTDataset] Train: {len(train_ds)} videos | Val: {len(val_ds)} videos | Test: {len(test_ds)} videos")
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_ds, val_ds, test_ds = build_mvit_datasets()
    clip, label = train_ds[0]
    print("Clip shape:", clip.shape, "| label:", label)
