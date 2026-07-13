"""
scripts/dataloader_builder.py
==============================
Step 3 (final) dari pipeline preprocessing:
  Augmented Sequences → Split Train/Val/Test → Simpan ke .npz / PyTorch Dataset

Output:
  - data/processed/<dataset>/dataloader/train.npz
  - data/processed/<dataset>/dataloader/val.npz
  - data/processed/<dataset>/dataloader/test.npz
  - data/processed/<dataset>/dataloader/label_map.json
  - data/processed/<dataset>/dataloader/split_info.json
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Split Strategy

def split_manifest(
    augmented_manifest: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    seed: int = 42,
    split_by_signer: bool = False,  # True = signer tertentu masuk ke test (lebih realistis)
    test_signers: Optional[List[int]] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split manifest menjadi train/val/test.

    Mode 1 (default): Random split berdasarkan rasio
    Mode 2: Split berdasarkan signer (lebih realistis, test = unseen signer)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Rasio train+val+test harus = 1.0"

    # Filter hanya original (tidak augment) untuk val dan test
    # Augmented data HANYA masuk ke train
    originals = [x for x in augmented_manifest if not x.get("is_augmented", False)]
    augmented = [x for x in augmented_manifest if x.get("is_augmented", False)]

    if split_by_signer and test_signers is not None:
        # Mode signer-based split
        test_data = [x for x in originals if x["signer_idx"] in test_signers]
        remaining = [x for x in originals if x["signer_idx"] not in test_signers]

        rng = np.random.default_rng(seed)
        if shuffle:
            rng.shuffle(remaining)

        n_val = int(len(remaining) * val_ratio / (train_ratio + val_ratio))
        val_data = remaining[:n_val]
        train_originals = remaining[n_val:]

        logger.info(f"Split by signer: test_signers={test_signers}")

    else:
        # Mode random split (per label, stratified)
        # Group by label
        by_label = defaultdict(list)
        for item in originals:
            by_label[item["label_idx"]].append(item)

        train_originals, val_data, test_data = [], [], []

        rng = np.random.default_rng(seed)

        for label_idx, items in by_label.items():
            items = list(items)
            if shuffle:
                rng.shuffle(items)

            n = len(items)
            n_test = max(1, int(n * test_ratio))
            n_val  = max(1, int(n * val_ratio))
            n_train = n - n_test - n_val

            train_originals += items[:n_train]
            val_data        += items[n_train:n_train + n_val]
            test_data       += items[n_train + n_val:]

    # Filter augmented agar hanya menyertakan augmentasi dari train_originals (mencegah kebocoran data)
    train_video_names = {x["video_name"] for x in train_originals}
    train_augmented = [x for x in augmented if x["video_name"] in train_video_names]
    train_data = train_originals + train_augmented

    if shuffle:
        rng = np.random.default_rng(seed + 1)
        rng.shuffle(train_data)

    logger.info(f"Split result:")
    logger.info(f"  Train : {len(train_data)} ({len(train_originals)} orig + {len(train_augmented)} aug)")
    logger.info(f"  Val   : {len(val_data)}")
    logger.info(f"  Test  : {len(test_data)}")

    return train_data, val_data, test_data

# Save to .npz

def build_npz_from_manifest(
    manifest: List[Dict],
    split_name: str,
    output_dir: Path,
    sequence_key: str = "augmented_npy",
) -> Path:
    """
    Load semua sequence dari manifest dan simpan dalam satu file .npz.

    File .npz berisi:
        X: (N, T, H, W, C) uint8  — sequences
        y: (N,) int               — label indices
        meta: JSON string          — metadata per sample
    """
    X_list, y_list, meta_list = [], [], []

    logger.info(f"  Membangun {split_name}.npz dari {len(manifest)} samples...")

    for item in manifest:
        npy_path = item.get(sequence_key) or item.get("frames_npy")
        if not npy_path or not Path(npy_path).exists():
            logger.warning(f"File tidak ditemukan: {npy_path}, skip")
            continue

        seq = np.load(npy_path)  # (T, H, W, C)
        X_list.append(seq)
        y_list.append(item["label_idx"])
        meta_list.append({
            "video_name": item["video_name"],
            "signer_idx": item["signer_idx"],
            "label_idx": item["label_idx"],
            "label_name": item["label_name"],
            "version": item.get("version", "original"),
        })

    if not X_list:
        logger.error(f"Tidak ada data untuk split {split_name}!")
        return None

    X = np.array(X_list, dtype=np.uint8)   # (N, T, H, W, C)
    y = np.array(y_list, dtype=np.int32)    # (N,)

    out_path = output_dir / f"{split_name}.npz"
    np.savez_compressed(str(out_path), X=X, y=y, meta=json.dumps(meta_list))

    logger.info(f"  Tersimpan: {out_path}")
    logger.info(f"    Shape X: {X.shape} | dtype: {X.dtype}")
    logger.info(f"    Shape y: {y.shape} | classes: {len(np.unique(y))}")

    return out_path

# Main

def build_dataloader(
    augmented_manifest_path: str,
    output_dir: str,
    preprocessing_config: dict,
    dataset_config: dict,
    overwrite: bool = False,
) -> Dict:
    """
    Entry point: baca manifest augmented → split → simpan .npz + label map.

    Returns:
        dict berisi path train/val/test dan label_map
    """
    dl_cfg  = preprocessing_config["dataloader"]
    out_dir = Path(output_dir) / "dataloader"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cek apakah sudah ada
    if not overwrite and (out_dir / "train.npz").exists():
        logger.info("Dataloader sudah ada, skip. Set overwrite=True untuk rebuild.")
        return _load_existing_info(out_dir)

    # Load manifest
    with open(augmented_manifest_path) as f:
        augmented_manifest = json.load(f)

    # Split
    train_manifest, val_manifest, test_manifest = split_manifest(
        augmented_manifest=augmented_manifest,
        train_ratio=dl_cfg["train_ratio"],
        val_ratio=dl_cfg["val_ratio"],
        test_ratio=dl_cfg["test_ratio"],
        shuffle=dl_cfg["shuffle"],
        seed=dl_cfg["random_seed"],
    )

    # Build .npz per split
    train_path = build_npz_from_manifest(train_manifest, "train", out_dir)
    val_path   = build_npz_from_manifest(val_manifest,   "val",   out_dir)
    test_path  = build_npz_from_manifest(test_manifest,  "test",  out_dir)

    # Buat label map
    label_map = _build_label_map(augmented_manifest, dataset_config)
    label_map_path = out_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    # Simpan split info
    split_info = {
        "train_count": len(train_manifest),
        "val_count": len(val_manifest),
        "test_count": len(test_manifest),
        "num_classes": len(label_map),
        "sequence_length": preprocessing_config["sequence"]["fixed_length"],
        "frame_size": preprocessing_config["frame"]["resize"],
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
        "label_map_path": str(label_map_path),
    }
    with open(out_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"Dataloader berhasil dibangun!")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"  Kelas : {len(label_map)}")
    logger.info(f"  Seq len: {preprocessing_config['sequence']['fixed_length']}")
    logger.info(f"{'='*50}\n")

    return split_info


def load_split(npz_path: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Helper untuk load .npz di notebook training.

    Returns:
        X    : (N, T, H, W, C) uint8
        y    : (N,) int
        meta : list of dict
    """
    data = np.load(npz_path, allow_pickle=True)
    X    = data["X"]
    y    = data["y"]
    meta = json.loads(str(data["meta"]))
    return X, y, meta

# Helpers

def _build_label_map(manifest: List[Dict], dataset_config: dict) -> Dict:
    """Buat label_map: {label_idx (str): label_name}"""
    manual_map = dataset_config.get("label_map")
    if manual_map:
        return {str(k): v for k, v in manual_map.items()}

    # Auto dari manifest
    label_map = {}
    for item in manifest:
        idx = item["label_idx"]
        name = item.get("label_name", f"label_{idx}")
        label_map[str(idx)] = name

    return dict(sorted(label_map.items(), key=lambda x: int(x[0])))


def _load_existing_info(out_dir: Path) -> Dict:
    info_path = out_dir / "split_info.json"
    if info_path.exists():
        with open(info_path) as f:
            return json.load(f)
    return {}