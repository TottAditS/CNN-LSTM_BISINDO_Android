"""
scripts/augmentor.py
=====================
Step 2 dari pipeline preprocessing:
  Frames → Frame Transforms → Frame Sampling → Padding → Augmented Sequences

Input : manifest dari frame_extractor (frame_manifest.json)
Output: augmented sequences, update manifest → augmented_manifest.json
"""

import numpy as np
import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Frame-level Transforms (diterapkan ke SETIAP frame dalam sequence)

def apply_horizontal_flip(frames: np.ndarray) -> np.ndarray:
    """Flip semua frame secara horizontal (mirror)."""
    return frames[:, :, ::-1, :].copy()


def apply_brightness(frames: np.ndarray, factor: float) -> np.ndarray:
    """Ubah brightness. factor: 0.7 (gelap) - 1.3 (terang)."""
    result = np.clip(frames.astype(np.float32) * factor, 0, 255)
    return result.astype(np.uint8)


def apply_contrast(frames: np.ndarray, factor: float) -> np.ndarray:
    """Ubah contrast. factor: 0.8 - 1.2."""
    mean = frames.mean(axis=(1, 2), keepdims=True)
    result = np.clip((frames.astype(np.float32) - mean) * factor + mean, 0, 255)
    return result.astype(np.uint8)


def apply_rotation(frames: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotasi setiap frame. Memerlukan cv2."""
    import cv2
    H, W = frames.shape[1], frames.shape[2]
    center = (W // 2, H // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = np.array([
        cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        for f in frames
    ], dtype=np.uint8)
    return rotated


def apply_zoom(frames: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Zoom in/out ke tengah frame."""
    import cv2
    H, W = frames.shape[1], frames.shape[2]
    new_H = int(H / zoom_factor)
    new_W = int(W / zoom_factor)
    start_y = (H - new_H) // 2
    start_x = (W - new_W) // 2

    zoomed = []
    for f in frames:
        crop = f[start_y:start_y+new_H, start_x:start_x+new_W]
        resized = cv2.resize(crop, (W, H))
        zoomed.append(resized)
    return np.array(zoomed, dtype=np.uint8)


def augment_sequence(
    frames: np.ndarray,
    techniques: List[str],
    config: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Terapkan augmentasi random ke satu sequence frames.

    Args:
        frames     : (N, H, W, C) uint8
        techniques : list teknik yang diaktifkan
        config     : augmentation config dari dataset_config.py
        rng        : numpy random generator (untuk reproducibility)

    Returns:
        np.ndarray (N, H, W, C) yang sudah di-augment
    """
    aug = frames.copy()

    if "horizontal_flip" in techniques:
        if rng.random() < config.get("horizontal_flip_prob", 0.5):
            aug = apply_horizontal_flip(aug)

    if "brightness" in techniques:
        lo, hi = config.get("brightness_range", (0.7, 1.3))
        factor = rng.uniform(lo, hi)
        aug = apply_brightness(aug, factor)

    if "contrast" in techniques:
        lo, hi = config.get("contrast_range", (0.8, 1.2))
        factor = rng.uniform(lo, hi)
        aug = apply_contrast(aug, factor)

    if "rotation" in techniques:
        lo, hi = config.get("rotation_range", (-10, 10))
        angle = rng.uniform(lo, hi)
        aug = apply_rotation(aug, angle)

    if "zoom" in techniques:
        lo, hi = config.get("zoom_range", (0.9, 1.1))
        factor = rng.uniform(lo, hi)
        aug = apply_zoom(aug, factor)

    return aug

# Sequence-level: Padding & Truncation

def standardize_sequence(
    frames: np.ndarray,
    fixed_length: int,
    padding_mode: str = "zero",
    truncate_mode: str = "center",
) -> np.ndarray:
    """
    Padatkan/potong sequence ke panjang fixed_length.

    Args:
        frames        : (N, H, W, C)
        fixed_length  : target jumlah frame
        padding_mode  : 'zero' atau 'repeat_last'
        truncate_mode : 'start', 'end', 'center'

    Returns:
        (fixed_length, H, W, C)
    """
    N = len(frames)
    H, W, C = frames.shape[1], frames.shape[2], frames.shape[3]

    if N == fixed_length:
        return frames

    elif N > fixed_length:
        # Truncate
        if truncate_mode == "start":
            return frames[:fixed_length]
        elif truncate_mode == "end":
            return frames[N - fixed_length:]
        else:  # center
            start = (N - fixed_length) // 2
            return frames[start:start + fixed_length]

    else:
        # Pad
        pad_count = fixed_length - N
        if padding_mode == "repeat_last":
            pad_frames = np.repeat(frames[-1:], pad_count, axis=0)
        else:  # zero
            pad_frames = np.zeros((pad_count, H, W, C), dtype=frames.dtype)
        return np.concatenate([frames, pad_frames], axis=0)

# Main Processing

def process_augmentation(
    frame_manifest_path: str,
    output_dir: str,
    preprocessing_config: dict,
    overwrite: bool = False,
) -> List[Dict]:
    """
    Baca manifest hasil frame_extractor, augment semua sequence, simpan hasilnya.

    Returns:
        List manifest augmented
    """
    aug_cfg = preprocessing_config["augmentation"]
    seq_cfg = preprocessing_config["sequence"]

    rng = np.random.default_rng(42)  # reproducible

    with open(frame_manifest_path) as f:
        manifest = json.load(f)

    out_dir = Path(output_dir) / "augmented"
    out_dir.mkdir(parents=True, exist_ok=True)

    augmented_manifest = []
    total = len(manifest)

    logger.info(f"Mulai augmentasi {total} sequences...")
    logger.info(f"  Teknik aktif: {aug_cfg['techniques']}")
    logger.info(f"  Multiplier  : {aug_cfg['augment_multiplier']}x")
    logger.info(f"  Fixed length: {seq_cfg['fixed_length']} frame")

    for i, item in enumerate(manifest):
        # Load frames
        frames = np.load(item["frames_npy"])  # (N, H, W, C)

        # Proses: original + N augmentasi
        versions = [("original", frames)]

        if aug_cfg["enabled"]:
            for aug_idx in range(aug_cfg["augment_multiplier"]):
                aug_frames = augment_sequence(
                    frames=frames,
                    techniques=aug_cfg["techniques"],
                    config=aug_cfg,
                    rng=rng,
                )
                versions.append((f"aug_{aug_idx}", aug_frames))

        for version_name, seq in versions:
            # Standardize (pad/truncate) ke fixed_length
            std_seq = standardize_sequence(
                frames=seq,
                fixed_length=seq_cfg["fixed_length"],
                padding_mode=seq_cfg["padding_mode"],
                truncate_mode=seq_cfg["truncate_mode"],
            )

            # Simpan
            video_stem = Path(item["video_name"]).stem
            out_subdir = out_dir / f"label_{item['label_idx']}"
            out_subdir.mkdir(parents=True, exist_ok=True)
            npy_path = out_subdir / f"{video_stem}_{version_name}.npy"

            if not npy_path.exists() or overwrite:
                np.save(str(npy_path), std_seq)

            augmented_manifest.append({
                **item,
                "augmented_npy": str(npy_path),
                "version": version_name,
                "is_augmented": version_name != "original",
                "sequence_length": seq_cfg["fixed_length"],
            })

        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i+1}/{total}")

    # Simpan manifest augmented
    aug_manifest_path = Path(output_dir) / "augmented_manifest.json"
    with open(aug_manifest_path, "w") as f:
        json.dump(augmented_manifest, f, indent=2)

    logger.info(f"\nAugmentasi selesai!")
    logger.info(f"  Total sequences: {len(augmented_manifest)}")
    logger.info(f"  Manifest: {aug_manifest_path}")

    return augmented_manifest