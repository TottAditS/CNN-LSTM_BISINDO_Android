"""
scripts/frame_extractor.py
===========================
Step 1 dari pipeline preprocessing:
  Video -> Extract Frames -> Blur Filter -> Resize -> Uniform Sampling -> Video Cutting

Output: folder berisi frame-frame .jpg per video
"""

import cv2
import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Core Functions

def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    target_fps: int = 15,
    resize: Tuple[int, int] = (224, 224),
    blur_kernel: Tuple[int, int] = (5, 5),
    blur_sigma: int = 0,
    max_frames: int = 60,
    min_frames: int = 10,
    save_as_images: bool = False,
) -> Optional[np.ndarray]:
    """
    Ekstrak frame dari satu video.

    Args:
        video_path     : Path ke file .mp4
        output_dir     : Folder output untuk menyimpan frame (jika save_as_images=True)
        target_fps     : FPS target setelah uniform sampling
        resize         : (H, W) ukuran frame setelah resize
        blur_kernel    : Kernel Gaussian blur
        blur_sigma     : Sigma blur (0 = auto)
        max_frames     : Maks jumlah frame yang diambil
        min_frames     : Min jumlah frame; return None jika video terlalu pendek
        save_as_images : Simpan tiap frame sebagai .jpg (untuk debug)

    Returns:
        np.ndarray shape (N, H, W, C) atau None jika gagal
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Tidak bisa membuka video: {video_path}")
        return None

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if original_fps <= 0:
        original_fps = 30.0  # fallback

    # Hitung interval sampling untuk uniform sampling
    sample_interval = max(1, round(original_fps / target_fps))

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Uniform sampling: ambil setiap N frame
        if frame_idx % sample_interval == 0:
            # 1. Blur filtering (noise reduction)
            frame = cv2.GaussianBlur(frame, blur_kernel, blur_sigma)

            # 2. Resize
            frame = cv2.resize(frame, (resize[1], resize[0]))  # cv2: (W, H)

            # 3. Convert BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)

            # Stop jika sudah max_frames
            if len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()

    if len(frames) < min_frames:
        logger.warning(
            f"Video terlalu pendek ({len(frames)} frame): {video_path}"
        )
        return None

    frames_array = np.array(frames, dtype=np.uint8)  # (N, H, W, C)

    # Simpan sebagai gambar jika diminta (mode debug)
    if save_as_images:
        os.makedirs(output_dir, exist_ok=True)
        for i, f in enumerate(frames):
            img_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}.jpg"), img_bgr)

    return frames_array


def process_dataset(
    raw_video_dir: str,
    output_dir: str,
    dataset_config: dict,
    preprocessing_config: dict,
    overwrite: bool = False,
) -> Dict:
    """
    Proses seluruh dataset: scan semua video, extract frames, simpan hasil.

    Returns:
        dict berisi statistik hasil preprocessing
    """
    raw_dir = Path(raw_video_dir)
    out_dir = Path(output_dir) / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_cfg = preprocessing_config["frame"]
    dl_cfg = preprocessing_config["dataloader"]

    # Scan semua video
    video_files = sorted(raw_dir.rglob("*.mp4"))
    logger.info(f"Ditemukan {len(video_files)} video di {raw_dir}")

    stats = {
        "total_videos": len(video_files),
        "processed": 0,
        "skipped_short": 0,
        "skipped_error": 0,
        "frame_counts": [],
        "label_distribution": {},
        "manifest": [],  # [{video, label, label_idx, frames_path, num_frames}]
    }

    for i, video_path in enumerate(video_files):
        # Parse nama file: signer0_label0_sample1.mp4
        stem = video_path.stem  # "signer0_label0_sample1"
        parts = _parse_filename(stem)

        if parts is None:
            logger.warning(f"Format nama file tidak dikenali: {video_path.name}")
            stats["skipped_error"] += 1
            continue

        label_idx = parts["label_idx"]
        label_name = _get_label_name(label_idx, dataset_config.get("label_map"))

        # Folder output per video: frames/label_0/signer0_label0_sample1/
        video_out_dir = out_dir / f"label_{label_idx}" / stem

        # Skip jika sudah ada dan overwrite=False
        meta_path = video_out_dir / "meta.json"
        if meta_path.exists() and not overwrite:
            # Load existing meta
            with open(meta_path) as f:
                meta = json.load(f)
            stats["manifest"].append(meta)
            stats["processed"] += 1
            stats["label_distribution"][label_name] = (
                stats["label_distribution"].get(label_name, 0) + 1
            )
            continue

        # Extract frames
        frames = extract_frames_from_video(
            video_path=str(video_path),
            output_dir=str(video_out_dir / "images"),
            target_fps=frame_cfg["target_fps"],
            resize=tuple(frame_cfg["resize"]),
            blur_kernel=tuple(frame_cfg["blur_kernel"]),
            blur_sigma=frame_cfg["blur_sigma"],
            max_frames=frame_cfg["max_frames"],
            min_frames=frame_cfg["min_frames"],
            save_as_images=dl_cfg.get("save_frames_as_images", False),
        )

        if frames is None:
            stats["skipped_short"] += 1
            continue

        # Simpan frames sebagai .npy
        video_out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = video_out_dir / "frames.npy"
        np.save(str(npy_path), frames)

        # Simpan metadata
        meta = {
            "video_name": video_path.name,
            "video_path": str(video_path),
            "frames_npy": str(npy_path),
            "label_idx": label_idx,
            "label_name": label_name,
            "signer_idx": parts["signer_idx"],
            "sample_idx": parts["sample_idx"],
            "num_frames": len(frames),
            "frame_shape": list(frames.shape[1:]),  # [H, W, C]
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        stats["manifest"].append(meta)
        stats["processed"] += 1
        stats["frame_counts"].append(len(frames))
        stats["label_distribution"][label_name] = (
            stats["label_distribution"].get(label_name, 0) + 1
        )

        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i+1}/{len(video_files)} video diproses")

    # Simpan manifest ke JSON
    manifest_path = Path(output_dir) / "frame_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(stats["manifest"], f, indent=2)

    # Summary statistik
    if stats["frame_counts"]:
        stats["avg_frames"] = float(np.mean(stats["frame_counts"]))
        stats["min_frames_found"] = int(np.min(stats["frame_counts"]))
        stats["max_frames_found"] = int(np.max(stats["frame_counts"]))
    else:
        stats["avg_frames"] = 0

    logger.info(f"\n{'='*50}")
    logger.info(f"Frame Extraction Selesai!")
    logger.info(f"  Diproses  : {stats['processed']}")
    logger.info(f"  Skip pendek: {stats['skipped_short']}")
    logger.info(f"  Skip error : {stats['skipped_error']}")
    if stats["frame_counts"]:
        logger.info(f"  Rata-rata frame: {stats['avg_frames']:.1f}")
        logger.info(f"  Frame range: {stats['min_frames_found']} - {stats['max_frames_found']}")
    logger.info(f"  Manifest disimpan: {manifest_path}")
    logger.info(f"{'='*50}\n")

    return stats

# Helpers

def _parse_filename(stem: str) -> Optional[Dict]:
    """
    Parse: signer0_label0_sample1 -> {signer_idx: 0, label_idx: 0, sample_idx: 1}
    Modular: tambahkan format lain di sini jika dataset berbeda.
    """
    try:
        parts = stem.split("_")
        signer_idx = int(parts[0].replace("signer", ""))
        label_idx  = int(parts[1].replace("label", ""))
        sample_idx = int(parts[2].replace("sample", ""))
        return {"signer_idx": signer_idx, "label_idx": label_idx, "sample_idx": sample_idx}
    except Exception:
        return None


def _get_label_name(label_idx: int, label_map: Optional[dict]) -> str:
    if label_map and label_idx in label_map:
        return label_map[label_idx]
    return f"label_{label_idx}"