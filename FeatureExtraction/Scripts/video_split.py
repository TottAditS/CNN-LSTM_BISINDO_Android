"""
FeatureExtraction/Scripts/video_split.py
==========================================
Split video (bukan sample) ke train/val/test, stratified per label — dipakai
BERSAMA oleh dua pipeline yang beda modalitas data tapi harus dibandingkan di
video val/test yang SAMA PERSIS:
- FeatureExtraction/Scripts/sequence_builder.py  (keypoint, untuk CNNsa-BiLSTM)
- Training/MViTv2/Scripts/mvit_dataset.py         (raw video frame, untuk MViTv2)

Kedua pipeline memanggil stratified_video_split() dengan seed & rasio yang sama
(dari FeatureExtraction/Config/feature_config.py), jadi video yang jadi val/test
untuk CNNsa-BiLSTM juga video yang sama persis jadi val/test untuk MViTv2 —
supaya perbandingan akurasi kedua model apple-to-apple.
"""

from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np


def stratified_video_split(
    manifest: List[Dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split video-level (satu manifest item = satu video), stratified per label_idx.

    Args:
        manifest   : list item dari frame_manifest.json (tiap item punya "label_idx")
        train/val/test_ratio: proporsi split, harus jumlah ~1.0
        seed       : random seed (RandomState) untuk reproducibility

    Returns:
        (train_items, val_items, test_items) — masing-masing subset dari manifest,
        urutan asli tiap item dipertahankan.
    """
    rng = np.random.RandomState(seed)

    label_to_indices = defaultdict(list)
    for i, item in enumerate(manifest):
        label_to_indices[item["label_idx"]].append(i)

    train_idx, val_idx, test_idx = [], [], []

    for label_idx, indices in label_to_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        # Sisanya masuk test

        train_idx.extend(indices[:n_train].tolist())
        val_idx.extend(indices[n_train : n_train + n_val].tolist())
        test_idx.extend(indices[n_train + n_val :].tolist())

    train_items = [manifest[i] for i in train_idx]
    val_items = [manifest[i] for i in val_idx]
    test_items = [manifest[i] for i in test_idx]

    return train_items, val_items, test_items
