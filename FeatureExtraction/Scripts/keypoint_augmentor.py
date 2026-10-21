"""
Augmentation of keypoint sequences (T, 126), used by sequence_builder.py.

The model reads normalized keypoints, so photometric augmentation has no
effect. The augmentations here change the keypoint values or timing instead:
geometric (flip, rotation, zoom), detection noise (jitter), signing speed
(temporal_speed), short detection failures (hand_dropout), and Signer-Mixup.

Layout of one frame (126,): [left hand (63), right hand (63)], each hand being
21 landmarks x (x, y, z). A hand that was not detected is all zeros, and the
operations below leave such blocks at zero.
"""

import numpy as np
from typing import List

KEYPOINTS_PER_HAND = 21
COORDS_PER_KEYPOINT = 3
HAND_BLOCK = KEYPOINTS_PER_HAND * COORDS_PER_KEYPOINT  # 63


def apply_horizontal_flip(kpts: np.ndarray) -> np.ndarray:
    """
    Mirror the sequence: negate every x coordinate and swap the left and right
    hand blocks.

    Args:
        kpts: (T, 126) float32
    """
    flipped = kpts.copy()
    flipped[:, 0::3] *= -1  # x coordinates sit at every third index in both hand blocks
    left, right = flipped[:, :HAND_BLOCK].copy(), flipped[:, HAND_BLOCK:].copy()
    flipped[:, :HAND_BLOCK] = right
    flipped[:, HAND_BLOCK:] = left
    return flipped


def _rotate_hand_block(block: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate (x, y) of each landmark around the wrist, which is the origin after normalization."""
    T = block.shape[0]
    pts = block.reshape(T, KEYPOINTS_PER_HAND, COORDS_PER_KEYPOINT).copy()
    x, y = pts[:, :, 0].copy(), pts[:, :, 1].copy()
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    pts[:, :, 0] = x * cos_a - y * sin_a
    pts[:, :, 1] = x * sin_a + y * cos_a
    return pts.reshape(T, HAND_BLOCK)


def apply_rotation(kpts: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate both hands around their own wrists. z is unchanged."""
    angle_rad = np.deg2rad(angle_deg)
    rotated = kpts.copy()
    rotated[:, :HAND_BLOCK] = _rotate_hand_block(kpts[:, :HAND_BLOCK], angle_rad)
    rotated[:, HAND_BLOCK:] = _rotate_hand_block(kpts[:, HAND_BLOCK:], angle_rad)
    return rotated


def apply_zoom(kpts: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Scale the distance of each landmark from the wrist."""
    return kpts * zoom_factor


def _hand_present_mask(kpts: np.ndarray, block_start: int) -> np.ndarray:
    """(T,) bool, True in frames where the given hand block is not all zeros."""
    block = kpts[:, block_start:block_start + HAND_BLOCK]
    return np.any(block != 0, axis=1)


def apply_landmark_jitter(kpts: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add small Gaussian noise to every coordinate to imitate MediaPipe detection
    noise under different cameras and lighting.

    Frames where a hand is missing are left at zero; otherwise the noise would
    turn "no hand" into a faint, fake detection.
    """
    jittered = kpts.copy()
    noise = rng.normal(0, noise_std, size=kpts.shape).astype(np.float32)
    for block_start in (0, HAND_BLOCK):
        mask = _hand_present_mask(kpts, block_start)
        jittered[mask, block_start:block_start + HAND_BLOCK] += noise[mask, block_start:block_start + HAND_BLOCK]
    return jittered


def apply_temporal_speed_variation(kpts: np.ndarray, speed_factor: float) -> np.ndarray:
    """
    Resample the sequence in time. speed_factor > 1 speeds the sign up (fewer
    frames), < 1 slows it down (more frames). The output length may differ from
    the input; pad_or_crop_sequence() in sequence_builder.py fixes it to
    SEQUENCE_LENGTH afterwards.
    """
    T = kpts.shape[0]
    new_T = max(1, int(round(T / speed_factor)))
    indices = np.linspace(0, T - 1, new_T).round().astype(int)
    return kpts[indices]


def apply_hand_dropout(
    kpts: np.ndarray,
    dropout_prob: float,
    max_frames_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    With probability dropout_prob, checked separately for each hand, zero out a
    run of consecutive frames of that hand (random length, at most
    max_frames_frac of the sequence). This imitates short detection failures
    from occlusion or fast motion. Only part of the sequence is dropped, so the
    hand still takes part in the sign.
    """
    dropped = kpts.copy()
    T = kpts.shape[0]
    for block_start in (0, HAND_BLOCK):
        if not _hand_present_mask(kpts, block_start).any():
            continue  # this hand never appears in the original sequence
        if rng.random() >= dropout_prob:
            continue
        drop_len = int(rng.integers(1, max(2, int(T * max_frames_frac) + 1)))
        start = int(rng.integers(0, max(1, T - drop_len + 1)))
        dropped[start:start + drop_len, block_start:block_start + HAND_BLOCK] = 0.0
    return dropped


def mix_sequences(kpts_a: np.ndarray, kpts_b: np.ndarray, alpha: float) -> np.ndarray:
    """
    Signer-Mixup: linear interpolation between two sequences of the same class
    recorded by different signers. The result lies between the two signers'
    styles, which adds signer variation that single-sequence augmentation cannot.

    kpts_b is first resampled to the length of kpts_a (nearest index via
    linspace) so the two can be mixed element-wise.

    Args:
        kpts_a, kpts_b: (T_a, 126) and (T_b, 126) float32, same class, different signers
        alpha: mixing weight in [0, 1]; callers keep it away from 0 and 1 (e.g. 0.3-0.7)

    Returns:
        (T_a, 126) float32
    """
    T_a = kpts_a.shape[0]
    T_b = kpts_b.shape[0]
    if T_b != T_a:
        indices = np.linspace(0, T_b - 1, T_a).round().astype(int)
        kpts_b = kpts_b[indices]
    return (alpha * kpts_a + (1.0 - alpha) * kpts_b).astype(np.float32)


def augment_sequence(
    kpts: np.ndarray,
    techniques: List[str],
    config: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply random augmentation to one sequence. Spatial transforms run first,
    then temporal_speed (which changes T), and hand_dropout last so its length
    is based on the final T.

    Args:
        kpts       : (T, 126) float32
        techniques : enabled technique names
        config     : AUGMENTATION_CONFIG from feature_config.py
        rng        : numpy random generator

    Returns:
        (T', 126) array; T' == T unless "temporal_speed" is enabled
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

    if "jitter" in techniques:
        noise_std = config.get("jitter_std", 0.02)
        aug = apply_landmark_jitter(aug, noise_std, rng)

    if "temporal_speed" in techniques:
        lo, hi = config.get("temporal_speed_range", (0.85, 1.15))
        aug = apply_temporal_speed_variation(aug, rng.uniform(lo, hi))

    if "hand_dropout" in techniques:
        prob = config.get("hand_dropout_prob", 0.15)
        max_frac = config.get("hand_dropout_max_frac", 0.3)
        aug = apply_hand_dropout(aug, prob, max_frac, rng)

    return aug
