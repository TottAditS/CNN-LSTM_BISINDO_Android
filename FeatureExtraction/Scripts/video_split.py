"""
Video-level train/val/test split, stratified by label.

Shared by both pipelines so that CNNsa-BiLSTM (keypoints, sequence_builder.py)
and MViTv2 (RGB frames, mvit_dataset.py) are evaluated on exactly the same
validation and test videos. Both call stratified_video_split() with the seed
and ratios from FeatureExtraction/Config/feature_config.py.
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
    Split manifest items (one item = one video), stratified by label_idx.

    Args:
        manifest   : items from frame_manifest.json, each with "label_idx"
        train/val/test_ratio: split proportions, summing to about 1.0
        seed       : RandomState seed

    Returns:
        (train_items, val_items, test_items), subsets of the manifest
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
        # the remainder goes to test

        train_idx.extend(indices[:n_train].tolist())
        val_idx.extend(indices[n_train : n_train + n_val].tolist())
        test_idx.extend(indices[n_train + n_val :].tolist())

    train_items = [manifest[i] for i in train_idx]
    val_items = [manifest[i] for i in val_idx]
    test_items = [manifest[i] for i in test_idx]

    return train_items, val_items, test_items
