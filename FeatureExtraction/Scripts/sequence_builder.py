"""
FeatureExtraction/Scripts/sequence_builder.py

Step 2: Split video ke train/val/test, augmentasi (khusus train), lalu
susun keypoint menjadi sequence fixed-length dan simpan sebagai .npz.
- Input : keypoint .npy per video (output keypoint_extractor.py)
- Output: train_sequences.npz, val_sequences.npz, test_sequences.npz
           X shape: (N, SEQUENCE_LENGTH, FEATURES_PER_FRAME) = (N, 30, 126)
           y shape: (N,) int64 -- label index

Urutan proses per video (penting untuk mencegah data leakage):
1. Split video (bukan sample) ke train/val/test, stratified per label.
2. Augmentasi keypoint (flip/rotate/zoom) HANYA diterapkan ke video yang
   masuk train -- val/test selalu berisi keypoint asli (belum diaugmentasi).
3. Pad/crop tiap sequence (asli maupun augmented) ke SEQUENCE_LENGTH.

Strategi panjang sequence:
- Jika T > SEQUENCE_LENGTH : ambil T frame tengah (center crop temporal)
- Jika T < SEQUENCE_LENGTH : pad dengan zeros di akhir
- Jika T == SEQUENCE_LENGTH: langsung pakai
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


# --- TEMPORAL RESIZE ---------------------------------------------------------

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


# --- SPLIT BUILDER -----------------------------------------------------------

class SequenceBuilder:
    """
    Split video (level manifest, SEBELUM baca keypoint) ke train/val/test per label,
    load keypoint tiap split, augmentasi video train, lalu simpan sebagai .npz.

    Split strategy: stratified_video_split() dari FeatureExtraction/Scripts/video_split.py
    (seed = RANDOM_SEED) -- fungsi SHARED yang sama dipakai Training/MViTv2/Scripts/mvit_dataset.py
    supaya video val/test CNNsa-BiLSTM dan MViTv2 identik (perbandingan apple-to-apple).
    Split dilakukan pada MANIFEST (semua video mentah), bukan pada video yang keypoint-nya
    sudah berhasil diekstrak -- video dengan keypoint hilang/gagal baru difilter setelah split,
    supaya hasil split tidak bergantung pada keberhasilan MediaPipe (yang notabene spesifik ke
    pipeline ini, tidak relevan untuk pipeline video mentah MViTv2).

    Augmentasi keypoint (flip/rotate/zoom) HANYA diterapkan ke video yang masuk train -- val/test
    selalu berisi keypoint asli (belum diaugmentasi), tidak ada leakage augmented data.
    """

    def __init__(self):
        self.aug_rng = np.random.default_rng(RANDOM_SEED)

    def _load_manifest(self) -> List[Dict]:
        manifest_path = PROCESSED_DIR / "frame_manifest.json"
        if not manifest_path.exists():
            print(f"[SequenceBuilder] ERROR: Manifest tidak ditemukan di {manifest_path}")
            return []
        with open(manifest_path, "r") as f:
            return json.load(f)

    def _load_keypoints_for_items(self, items: List[Dict]) -> Tuple[List[np.ndarray], List[int], List[str], List[int]]:
        """
        Load file keypoint .npy untuk sekumpulan item manifest (satu split).
        Item yang keypoint-nya belum ada / gagal diekstrak MediaPipe di-skip (dengan warning).

        signer_id tiap item ikut dikembalikan (dulu tidak) -- dipakai _build_split() untuk
        Signer-Mixup, yang perlu tahu SIAPA signer tiap sample supaya bisa mencampur sample
        dari signer BERBEDA pada kelas yang sama.
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
                print(f"  [SKIP] Shape tidak valid {kpt_path}: {kpts.shape}")
                continue

            X_list.append(kpts)
            y_list.append(label_id)
            id_list.append(vid_name)
            signer_list.append(signer_id)

        if missing > 0:
            print(f"  [WARN] {missing} file keypoint tidak ditemukan (di-skip dari split ini).")

        return X_list, y_list, id_list, signer_list

    def _build_split(self, split_name: str, X_list: List[np.ndarray], y: List[int],
                      ids: List[str], signers: List[int]) -> Dict:
        """
        Bangun array X/y/ids final untuk satu split.
        Augmentasi (termasuk Signer-Mixup) hanya diterapkan jika split_name == "train".
        """
        aug_cfg = AUGMENTATION_CONFIG
        do_augment = split_name == "train" and aug_cfg.get("enabled", False)
        mixup_count = aug_cfg.get("signer_mixup_count", 0) if do_augment else 0

        # Lookup label -> signer -> list of kpts, dibangun SEKALI, dipakai Signer-Mixup untuk
        # cari kandidat "signer lain, kelas sama" tiap item train. Cuma perlu kalau mixup aktif.
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
                        break  # kelas ini di train split cuma ada 1 signer -- tidak ada pasangan
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
        """
        Pipeline lengkap: manifest -> split (video-level, shared) -> load keypoint per split
        -> augment (train only) -> resize -> save.
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

        # 1. Load manifest & split video-level (SEBELUM baca keypoint / augmentasi)
        manifest = self._load_manifest()
        if not manifest:
            print("[SequenceBuilder] ERROR: Manifest kosong/tidak ada.")
            print("  Pastikan PreProcessing/PreProcess.ipynb sudah dijalankan terlebih dahulu.")
            return

        print("[SequenceBuilder] Stratified split (video-level, shared dgn MViTv2)...")
        train_items, val_items, test_items = stratified_video_split(
            manifest, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
        )

        aug_cfg = AUGMENTATION_CONFIG
        print(f"[SequenceBuilder] Augmentasi: enabled={aug_cfg['enabled']}, "
              f"teknik={aug_cfg['techniques']}, multiplier={aug_cfg['augment_multiplier']}x, "
              f"signer_mixup_count={aug_cfg.get('signer_mixup_count', 0)}x "
              f"(hanya diterapkan ke train)")

        # 2. Load keypoint per split, lalu augment (train only) + temporal resize
        print("[SequenceBuilder] Loading keypoint files...")
        splits = {}
        for split_name, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
            X_list, y, ids, signers = self._load_keypoints_for_items(items)
            print(f"  [{split_name:5s}] {len(X_list)} video keypoint loaded")
            splits[split_name] = self._build_split(split_name, X_list, y, ids, signers)

        # 3. Simpan .npz per split
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

        print("\n[SequenceBuilder] Selesai!")
        self._print_stats()

    def _print_stats(self):
        """Print statistik dari .npz yang sudah ada."""
        print("\n--- Statistik Sequences ---------------------------------------")
        for split in ["train", "val", "test"]:
            path = get_sequence_output_path(split)
            if path.exists():
                data = np.load(path, allow_pickle=True)
                X, y = data["X"], data["y"]
                print(f"  {split:5s}: X={X.shape}, y={y.shape}, "
                      f"labels={np.unique(y).shape[0]}")
        print("-----------------------------------------------------------------")


# --- VERIFY ------------------------------------------------------------------

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
