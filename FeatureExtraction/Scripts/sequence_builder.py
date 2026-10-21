"""
Builds the fixed-length keypoint sequences used to train CNNsa-BiLSTM.

Input : keypoint .npy per video (keypoint_extractor.py output)
Output: train_sequences.npz, val_sequences.npz, test_sequences.npz
        X: (N, SEQUENCE_LENGTH, FEATURES_PER_FRAME) = (N, 30, 126)
        y: (N,) int64 label index

Order of operations, which prevents leakage between splits:
1. Split videos into train/val/test, stratified by label.
2. Augment the training videos only; val/test keep the original keypoints.
3. Pad or crop every sequence to SEQUENCE_LENGTH.

Sequence length:
- T > SEQUENCE_LENGTH : keep the middle frames (temporal center crop)
- T < SEQUENCE_LENGTH : pad with zeros at the end
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from FeatureExtraction.Config.feature_config import (
    PROCESSED_DIR, KEYPOINT_DIR, SEQUENCE_DIR,
    NUM_LABELS,
    SEQUENCE_LENGTH, FEATURES_PER_FRAME,
    CENTER_CROP_TEMPORAL, PAD_VALUE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
    AUGMENTATION_CONFIG,
    LABEL_MAP, VERBOSE,
    get_keypoint_path, get_sequence_output_path,
    create_output_dirs,
)
from FeatureExtraction.Scripts.keypoint_augmentor import augment_sequence, mix_sequences
from FeatureExtraction.Scripts.video_split import stratified_video_split


def pad_or_crop_sequence(
    keypoints: np.ndarray,
    target_length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """
    Resize a sequence in time to target_length.

    Args:
        keypoints    : (T, F) float32
        target_length: output length (default SEQUENCE_LENGTH = 30)

    Returns:
        sequence: (target_length, F) float32
    """
    T, F = keypoints.shape

    if T == target_length:
        return keypoints

    elif T > target_length:
        if CENTER_CROP_TEMPORAL:
            start = (T - target_length) // 2
            return keypoints[start : start + target_length]
        else:
            return keypoints[:target_length]

    else:
        pad_len = target_length - T
        pad = np.full((pad_len, F), PAD_VALUE, dtype=np.float32)
        return np.concatenate([keypoints, pad], axis=0)


class SequenceBuilder:
    """
    Splits the manifest into train/val/test, loads the keypoints of each split,
    augments the training videos, and saves each split as .npz.

    The split uses stratified_video_split() from video_split.py, the same
    function used by the MViTv2 dataset, so both models see identical val/test
    videos. The split is made on the full manifest; videos whose keypoint file
    is missing are dropped afterwards, so the split does not depend on MediaPipe.
    """

    def __init__(self):
        self.aug_rng = np.random.default_rng(RANDOM_SEED)

    def _load_manifest(self) -> List[Dict]:
        manifest_path = PROCESSED_DIR / "frame_manifest.json"
        if not manifest_path.exists():
            print(f"[SequenceBuilder] ERROR: manifest not found at {manifest_path}")
            return []
        with open(manifest_path, "r") as f:
            return json.load(f)

    def _load_keypoints_for_items(self, items: List[Dict]) -> Tuple[List[np.ndarray], List[int], List[str], List[int]]:
        """
        Load the keypoint files of one split. Items without a valid keypoint
        file are skipped with a warning. The signer of each item is returned
        for Signer-Mixup.
        """
        X_list, y_list, id_list, signer_list = [], [], [], []
        missing = 0

        for item in items:
            signer_id = item["signer_idx"]
            label_id  = item["label_idx"]
            sample_id = item["sample_idx"]

            kpt_path = get_keypoint_path(signer_id, label_id, sample_id)
            vid_name = Path(item["video_name"]).stem

            if not kpt_path.exists():
                missing += 1
                continue

            kpts = np.load(kpt_path)  # (T, 126)
            if kpts.ndim != 2 or kpts.shape[1] != FEATURES_PER_FRAME:
                print(f"  [SKIP] Invalid shape {kpt_path}: {kpts.shape}")
                continue

            X_list.append(kpts)
            y_list.append(label_id)
            id_list.append(vid_name)
            signer_list.append(signer_id)

        if missing > 0:
            print(f"  [WARN] {missing} keypoint files not found, skipped from this split.")

        return X_list, y_list, id_list, signer_list

    def _build_split(self, split_name: str, X_list: List[np.ndarray], y: List[int],
                      ids: List[str], signers: List[int]) -> Dict:
        """
        Build the final X/y/ids arrays of one split. Augmentation and
        Signer-Mixup are applied only when split_name == "train".
        """
        aug_cfg = AUGMENTATION_CONFIG
        do_augment = split_name == "train" and aug_cfg.get("enabled", False)
        mixup_count = aug_cfg.get("signer_mixup_count", 0) if do_augment else 0

        # label -> signer -> sequences, used to find same-class partners from other signers
        by_label_signer: Dict[int, Dict[int, List[np.ndarray]]] = {}
        if mixup_count > 0:
            for kpts, label, signer in zip(X_list, y, signers):
                by_label_signer.setdefault(label, {}).setdefault(signer, []).append(kpts)

        X_out, y_out, id_out = [], [], []

        for i in tqdm(range(len(X_list)), desc=f"Build {split_name}", unit="video"):
            kpts, label, vid_name = X_list[i], y[i], ids[i]

            versions = [("original", kpts)]
            if do_augment:
                for a in range(aug_cfg["augment_multiplier"]):
                    aug_kpts = augment_sequence(
                        kpts=kpts,
                        techniques=aug_cfg["techniques"],
                        config=aug_cfg,
                        rng=self.aug_rng,
                    )
                    versions.append((f"aug_{a}", aug_kpts))

            if mixup_count > 0:
                own_signer = signers[i]
                candidates_by_signer = by_label_signer.get(label, {})
                other_signers = [s for s in candidates_by_signer if s != own_signer]
                lo, hi = aug_cfg.get("signer_mixup_alpha_range", (0.3, 0.7))
                for m in range(mixup_count):
                    if not other_signers:
                        break  # only one signer has this class in the training split
                    other_signer = other_signers[self.aug_rng.integers(len(other_signers))]
                    pool = candidates_by_signer[other_signer]
                    other_kpts = pool[self.aug_rng.integers(len(pool))]
                    alpha = self.aug_rng.uniform(lo, hi)
                    mixed = mix_sequences(kpts, other_kpts, alpha)
                    versions.append((f"mixup_{m}", mixed))

            for version_name, seq in versions:
                X_out.append(pad_or_crop_sequence(seq, SEQUENCE_LENGTH))
                y_out.append(label)
                id_out.append(f"{vid_name}_{version_name}")

        return {
            "X"  : np.stack(X_out).astype(np.float32) if X_out else np.zeros((0, SEQUENCE_LENGTH, FEATURES_PER_FRAME), dtype=np.float32),
            "y"  : np.array(y_out, dtype=np.int64),
            "ids": np.array(id_out),
        }

    def build(self, skip_existing: bool = True):
        """Manifest -> split -> load keypoints -> augment (train only) -> resize -> save."""
        create_output_dirs()

        if skip_existing:
            all_exist = all(
                get_sequence_output_path(split).exists()
                for split in ["train", "val", "test"]
            )
            if all_exist:
                print("[SequenceBuilder] All sequence .npz files exist, skipping.")
                self._print_stats()
                return

        manifest = self._load_manifest()
        if not manifest:
            print("[SequenceBuilder] ERROR: manifest is empty or missing.")
            print("  Run PreProcessing/PreProcess.ipynb first.")
            return

        print("[SequenceBuilder] Stratified video-level split (shared with MViTv2)...")
        train_items, val_items, test_items = stratified_video_split(
            manifest, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
        )

        aug_cfg = AUGMENTATION_CONFIG
        print(f"[SequenceBuilder] Augmentation: enabled={aug_cfg['enabled']}, "
              f"techniques={aug_cfg['techniques']}, multiplier={aug_cfg['augment_multiplier']}x, "
              f"signer_mixup_count={aug_cfg.get('signer_mixup_count', 0)}x "
              f"(training split only)")

        print("[SequenceBuilder] Loading keypoint files...")
        splits = {}
        for split_name, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
            X_list, y, ids, signers = self._load_keypoints_for_items(items)
            print(f"  [{split_name:5s}] {len(X_list)} videos loaded")
            splits[split_name] = self._build_split(split_name, X_list, y, ids, signers)

        for split_name, data in splits.items():
            out_path = get_sequence_output_path(split_name)
            np.savez_compressed(
                out_path,
                X         = data["X"],
                y         = data["y"],
                ids       = data["ids"],
                label_map = np.array([LABEL_MAP[i] for i in range(NUM_LABELS)]),
            )
            n = len(data["y"])
            print(f"  [{split_name:5s}] {n:5d} sequences -> {out_path}")

        print("\n[SequenceBuilder] Done.")
        self._print_stats()

    def _print_stats(self):
        """Print the shapes of the saved .npz files."""
        print("\n--- Sequence statistics -----------------------------------------")
        for split in ["train", "val", "test"]:
            path = get_sequence_output_path(split)
            if path.exists():
                data = np.load(path, allow_pickle=True)
                X, y = data["X"], data["y"]
                print(f"  {split:5s}: X={X.shape}, y={y.shape}, "
                      f"labels={np.unique(y).shape[0]}")
        print("-----------------------------------------------------------------")


def verify_sequences():
    """Load the saved sequences and print basic statistics."""
    print("=== Sequence check ===\n")
    for split in ["train", "val", "test"]:
        path = get_sequence_output_path(split)
        if not path.exists():
            print(f"[{split}] File not found: {path}")
            continue

        data = np.load(path, allow_pickle=True)
        X, y = data["X"], data["y"]

        print(f"[{split}]")
        print(f"  X shape    : {X.shape}")
        print(f"  y shape    : {y.shape}")
        print(f"  X dtype    : {X.dtype}")
        print(f"  X min/max  : {X.min():.4f} / {X.max():.4f}")
        print(f"  y unique   : {np.unique(y)}")

        label_counts = np.bincount(y, minlength=NUM_LABELS)
        print(f"  Samples/label (min/max): {label_counts.min()} / {label_counts.max()}")

        zero_frames = np.all(X == 0, axis=-1)  # (N, T) bool
        zero_rate = zero_frames.mean()
        print(f"  Zero frames: {zero_rate*100:.1f}% "
              f"(no hand detected)")
        print()

    print("=== Check finished ===")


if __name__ == "__main__":
    builder = SequenceBuilder()
    builder.build(skip_existing=True)
    verify_sequences()
