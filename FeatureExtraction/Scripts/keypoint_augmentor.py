"""
FeatureExtraction/Scripts/keypoint_augmentor.py
=================================================
Augmentasi sequence keypoint (T, 126) — dipakai oleh sequence_builder.py.

Kenapa di sini, bukan di PreProcessing (pixel-level)?
Model final (CNNsa-BiLSTM) memakai keypoint MediaPipe, bukan raw pixel frame.
Augmentasi fotometrik (brightness/contrast) tidak berpengaruh ke koordinat
keypoint yang sudah dinormalisasi (relatif ke wrist, scale-invariant) — jadi
hanya transformasi geometris (flip, rotate, zoom) yang benar-benar mengubah
nilai keypoint dan berguna sebagai augmentasi.

Layout satu baris keypoint (126,): [tangan_kiri(63), tangan_kanan(63)],
tiap tangan = 21 keypoint x (x, y, z) berurutan. Tangan yang tidak
terdeteksi bernilai nol — semua operasi di bawah ini no-op pada nol.
"""

import numpy as np
from typing import List

KEYPOINTS_PER_HAND = 21
COORDS_PER_KEYPOINT = 3
HAND_BLOCK = KEYPOINTS_PER_HAND * COORDS_PER_KEYPOINT  # 63


def apply_horizontal_flip(kpts: np.ndarray) -> np.ndarray:
    """
    Mirror sequence secara horizontal: negasikan koordinat x tiap keypoint,
    lalu tukar blok tangan kiri <-> kanan (mencerminkan seluruh scene).

    Args:
        kpts: (T, 126) float32
    """
    flipped = kpts.copy()
    flipped[:, 0::3] *= -1  # x-coord ada di index kelipatan 3 (0,3,6,... di kedua blok tangan)
    left, right = flipped[:, :HAND_BLOCK].copy(), flipped[:, HAND_BLOCK:].copy()
    flipped[:, :HAND_BLOCK] = right
    flipped[:, HAND_BLOCK:] = left
    return flipped


def _rotate_hand_block(block: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotasi (x, y) tiap keypoint di sekitar wrist (origin, karena sudah dinormalisasi)."""
    T = block.shape[0]
    pts = block.reshape(T, KEYPOINTS_PER_HAND, COORDS_PER_KEYPOINT).copy()
    x, y = pts[:, :, 0].copy(), pts[:, :, 1].copy()
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    pts[:, :, 0] = x * cos_a - y * sin_a
    pts[:, :, 1] = x * sin_a + y * cos_a
    return pts.reshape(T, HAND_BLOCK)


def apply_rotation(kpts: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotasi kedua tangan di sekitar wrist masing-masing. z tidak berubah."""
    angle_rad = np.deg2rad(angle_deg)
    rotated = kpts.copy()
    rotated[:, :HAND_BLOCK] = _rotate_hand_block(kpts[:, :HAND_BLOCK], angle_rad)
    rotated[:, HAND_BLOCK:] = _rotate_hand_block(kpts[:, HAND_BLOCK:], angle_rad)
    return rotated


def apply_zoom(kpts: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Skala jarak tiap keypoint dari wrist — variasi ringan untuk regularisasi."""
    return kpts * zoom_factor


def augment_sequence(
    kpts: np.ndarray,
    techniques: List[str],
    config: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Terapkan augmentasi random ke satu sequence keypoint.

    Args:
        kpts       : (T, 126) float32
        techniques : list teknik yang diaktifkan
        config     : AUGMENTATION_CONFIG dari feature_config.py
        rng        : numpy random generator (untuk reproducibility)

    Returns:
        np.ndarray (T, 126) yang sudah di-augment
    """
    aug = kpts.copy()

    if "horizontal_flip" in techniques:
        if rng.random() < config.get("horizontal_flip_prob", 0.5):
            aug = apply_horizontal_flip(aug)

    if "rotation" in techniques:
        lo, hi = config.get("rotation_range", (-10, 10))
        aug = apply_rotation(aug, rng.uniform(lo, hi))

    if "zoom" in techniques:
        lo, hi = config.get("zoom_range", (0.9, 1.1))
        aug = apply_zoom(aug, rng.uniform(lo, hi))

    return aug
