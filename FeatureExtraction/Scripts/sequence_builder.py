"""
FeatureExtraction/Scripts/sequence_builder.py

Step 3: Susun keypoint menjadi sequence fixed-length dan simpan sebagai .npz.
- Input : keypoint .npy per video (output keypoint_extractor.py)
           split info dari PreProcessing (train/val/test)
- Output: train_sequences.npz, val_sequences.npz, test_sequences.npz
           X shape: (N, SEQUENCE_LENGTH, FEATURES_PER_FRAME) = (N, 30, 126)
           y shape: (N,) int64 — label index

Strategi panjang sequence:
- Jika T > SEQUENCE_LENGTH : ambil T frame tengah (center crop temporal)
- Jika T < SEQUENCE_LENGTH : pad dengan zeros di akhir
- Jika T == SEQUENCE_LENGTH: langsung pakai
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from FeatureExtraction.Config.feature_config import (
    PROCESSED_DIR, KEYPOINT_DIR, SEQUENCE_DIR,
    NUM_SIGNERS, NUM_LABELS, SAMPLES_PER_LABEL_PER_SIGNER,
    SEQUENCE_LENGTH, FEATURES_PER_FRAME,
    CENTER_CROP_TEMPORAL, PAD_VALUE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
    LABEL_MAP, VERBOSE,
    get_keypoint_path, get_sequence_output_path,
    create_output_dirs,
)


# ─── TEMPORAL RESIZE ─────────────────────────────────────────────────────────

def pad_or_crop_sequence(
    keypoints: np.ndarray,
    target_length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """
    Resize sequence temporal ke target_length.
    
    Args:
        keypoints    : (T, F) float32
        target_length: panjang target (default: SEQUENCE_LENGTH = 30)
        
    Returns:
        sequence: (target_length, F) float32
    """
    T, F = keypoints.shape

    if T == target_length:
        return keypoints

    elif T > target_length:
        # Center crop temporal: ambil frame dari tengah
        if CENTER_CROP_TEMPORAL:
            start = (T - target_length) // 2
            return keypoints[start : start + target_length]
        else:
            # Ambil target_length frame pertama
            return keypoints[:target_length]

    else:
        # Pad dengan PAD_VALUE di akhir
        pad_len = target_length - T
        pad = np.full((pad_len, F), PAD_VALUE, dtype=np.float32)
        return np.concatenate([keypoints, pad], axis=0)


# ─── SPLIT BUILDER ───────────────────────────────────────────────────────────

class SequenceBuilder:
    """
    Load semua keypoint .npy, bagi ke train/val/test per label,
    dan simpan sebagai .npz.
    
    Split strategy:
    - Split per label, stratified berdasarkan signer
    - signer0,1,2 majority ke train, signer3 ke val/test
    - Atau random split jika tidak mau signer-based
    
    Default: random split (TRAIN_RATIO / VAL_RATIO / TEST_RATIO)
    dengan seed = RANDOM_SEED — konsisten dengan PreProcessing.
    
    Catatan: Augmentasi sudah dilakukan di PreProcessing.
    Augmented data sudah masuk ke train split.
    Di sini kita hanya build sequences dari original (dan augmented jika ada).
    """

    def __init__(self):
        self.rng = np.random.RandomState(RANDOM_SEED)

    def _load_all_keypoints(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load semua keypoint yang tersedia.
        
        Returns:
            X   : (N, T_raw, 126) — sebelum temporal resize
            y   : (N,) int64 — label index
            ids : list of video identifier strings
        """
        X_list    = []
        y_list    = []
        id_list   = []
        missing   = 0

        print("[SequenceBuilder] Loading keypoint files...")

        for signer_id in range(NUM_SIGNERS):
            for label_id in range(NUM_LABELS):
                for sample_id in range(1, SAMPLES_PER_LABEL_PER_SIGNER + 1):

                    kpt_path = get_keypoint_path(signer_id, label_id, sample_id)
                    vid_name = f"signer{signer_id}_label{label_id}_sample{sample_id}"

                    if not kpt_path.exists():
                        missing += 1
                        continue

                    kpts = np.load(kpt_path)  # (T, 126)
                    if kpts.ndim != 2 or kpts.shape[1] != FEATURES_PER_FRAME:
                        print(f"  [SKIP] Shape tidak valid {kpt_path}: {kpts.shape}")
                        continue

                    X_list.append(kpts)
                    y_list.append(label_id)
                    id_list.append(vid_name)

        if missing > 0:
            print(f"  [WARN] {missing} file keypoint tidak ditemukan.")

        print(f"  Total sequence loaded: {len(X_list)}")
        return X_list, np.array(y_list, dtype=np.int64), id_list

    def _temporal_resize_all(self, X_list: List[np.ndarray]) -> np.ndarray:
        """
        Terapkan pad_or_crop_sequence ke semua sequence.
        
        Returns:
            X: (N, SEQUENCE_LENGTH, FEATURES_PER_FRAME) float32
        """
        N = len(X_list)
        X = np.zeros((N, SEQUENCE_LENGTH, FEATURES_PER_FRAME), dtype=np.float32)
        for i, kpts in enumerate(X_list):
            X[i] = pad_or_crop_sequence(kpts, SEQUENCE_LENGTH)
        return X

    def _stratified_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ids: List[str],
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Split data menjadi train / val / test secara stratified per label.
        
        Returns:
            train, val, test: masing-masing dict {"X": ..., "y": ..., "ids": ...}
        """
        train_idx, val_idx, test_idx = [], [], []

        label_to_indices = defaultdict(list)
        for i, label in enumerate(y):
            label_to_indices[label].append(i)

        for label, indices in label_to_indices.items():
            indices = np.array(indices)
            self.rng.shuffle(indices)
            n = len(indices)
            n_train = max(1, int(n * TRAIN_RATIO))
            n_val   = max(1, int(n * VAL_RATIO))
            # Sisanya masuk test

            train_idx.extend(indices[:n_train].tolist())
            val_idx.extend(indices[n_train : n_train + n_val].tolist())
            test_idx.extend(indices[n_train + n_val:].tolist())

        def make_split(idxs: List[int]) -> Dict:
            idxs = np.array(idxs)
            return {
                "X"   : X[idxs],
                "y"   : y[idxs],
                "ids" : [ids[i] for i in idxs],
            }

        return make_split(train_idx), make_split(val_idx), make_split(test_idx)

    def build(self, skip_existing: bool = True):
        """
        Pipeline lengkap: load → resize → split → save.
        """
        create_output_dirs()

        # Check apakah output sudah ada
        if skip_existing:
            all_exist = all(
                get_sequence_output_path(split).exists()
                for split in ["train", "val", "test"]
            )
            if all_exist:
                print("[SequenceBuilder] Semua sequence .npz sudah ada, skip.")
                self._print_stats()
                return

        # 1. Load semua keypoint
        X_list, y, ids = self._load_all_keypoints()
        if len(X_list) == 0:
            print("[SequenceBuilder] ERROR: Tidak ada keypoint yang berhasil diload!")
            print("  Pastikan keypoint_extractor.py sudah dijalankan terlebih dahulu.")
            return

        # 2. Temporal resize ke SEQUENCE_LENGTH
        print(f"[SequenceBuilder] Temporal resize ke {SEQUENCE_LENGTH} frame...")
        X = self._temporal_resize_all(X_list)
        print(f"  X shape: {X.shape}, dtype: {X.dtype}")
        print(f"  y shape: {y.shape}, unique labels: {np.unique(y).shape[0]}")

        # 3. Stratified split
        print("[SequenceBuilder] Stratified split...")
        train, val, test = self._stratified_split(X, y, ids)

        # 4. Simpan .npz per split
        for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
            out_path = get_sequence_output_path(split_name)
            np.savez_compressed(
                out_path,
                X    = split_data["X"],
                y    = split_data["y"],
                ids  = np.array(split_data["ids"]),
                label_map = np.array([LABEL_MAP[i] for i in range(NUM_LABELS)]),
            )
            n = len(split_data["y"])
            print(f"  [{split_name:5s}] {n:5d} sequences → {out_path}")

        print("\n[SequenceBuilder] ✅ Selesai!")
        self._print_stats()

    def _print_stats(self):
        """Print statistik dari .npz yang sudah ada."""
        print("\n─── Statistik Sequences ───────────────────────────────────────")
        for split in ["train", "val", "test"]:
            path = get_sequence_output_path(split)
            if path.exists():
                data = np.load(path, allow_pickle=True)
                X, y = data["X"], data["y"]
                print(f"  {split:5s}: X={X.shape}, y={y.shape}, "
                      f"labels={np.unique(y).shape[0]}")
        print("────────────────────────────────────────────────────────────────")


# ─── VERIFY ──────────────────────────────────────────────────────────────────

def verify_sequences():
    """
    Quick sanity check: load sequences dan tampilkan statistik.
    Dipakai dari notebook untuk verifikasi.
    """
    print("=== Verifikasi Sequences ===\n")
    for split in ["train", "val", "test"]:
        path = get_sequence_output_path(split)
        if not path.exists():
            print(f"[{split}] File tidak ditemukan: {path}")
            continue

        data = np.load(path, allow_pickle=True)
        X, y = data["X"], data["y"]

        print(f"[{split}]")
        print(f"  X shape    : {X.shape}")
        print(f"  y shape    : {y.shape}")
        print(f"  X dtype    : {X.dtype}")
        print(f"  X min/max  : {X.min():.4f} / {X.max():.4f}")
        print(f"  y unique   : {np.unique(y)}")

        # Cek distribusi label
        label_counts = np.bincount(y, minlength=NUM_LABELS)
        print(f"  Samples/label (min/max): {label_counts.min()} / {label_counts.max()}")

        # Cek zero frames (tangan tidak terdeteksi)
        zero_frames = np.all(X == 0, axis=-1)  # (N, T) bool
        zero_rate = zero_frames.mean()
        print(f"  Frame zero  : {zero_rate*100:.1f}% "
              f"(tangan tidak terdeteksi)")
        print()

    print("=== Verifikasi Selesai ===")


if __name__ == "__main__":
    builder = SequenceBuilder()
    builder.build(skip_existing=True)
    verify_sequences()
