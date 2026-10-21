"""
Preprocessing step: video -> uniformly sampled, blurred, resized RGB frames.

Each video is saved as frames.npy (N, H, W, 3) with a meta.json next to it, and
the metadata of all videos is collected in frame_manifest.json.
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
    Extract the frames of one video.

    Args:
        video_path     : path to the .mp4 file
        output_dir     : folder for .jpg frames (only used when save_as_images=True)
        target_fps     : frame rate after uniform sampling
        resize         : (H, W) of the output frames
        blur_kernel    : Gaussian blur kernel
        blur_sigma     : blur sigma (0 = derived from the kernel)
        max_frames     : maximum number of frames kept
        min_frames     : videos with fewer frames return None
        save_as_images : also write each frame as .jpg (debugging)

    Returns:
        np.ndarray of shape (N, H, W, C), or None if the video is unusable
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return None

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if original_fps <= 0:
        original_fps = 30.0

    sample_interval = max(1, round(original_fps / target_fps))

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Uniform sampling: keep every sample_interval-th frame
        if frame_idx % sample_interval == 0:
            frame = cv2.GaussianBlur(frame, blur_kernel, blur_sigma)
            frame = cv2.resize(frame, (resize[1], resize[0]))  # cv2 expects (W, H)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)

            if len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()

    if len(frames) < min_frames:
        logger.warning(
            f"Video too short ({len(frames)} frames): {video_path}"
        )
        return None

    frames_array = np.array(frames, dtype=np.uint8)  # (N, H, W, C)

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
    Extract frames for every video in raw_video_dir and write the manifest.

    Returns:
        dict with preprocessing statistics
    """
    raw_dir = Path(raw_video_dir)
    out_dir = Path(output_dir) / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_cfg = preprocessing_config["frame"]
    dl_cfg = preprocessing_config["dataloader"]

    video_files = sorted(raw_dir.rglob("*.mp4"))
    logger.info(f"Found {len(video_files)} videos in {raw_dir}")

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
        stem = video_path.stem  # e.g. "signer0_label0_sample1"
        parts = _parse_filename(stem)

        if parts is None:
            logger.warning(f"Unrecognized file name: {video_path.name}")
            stats["skipped_error"] += 1
            continue

        label_idx = parts["label_idx"]
        label_name = _get_label_name(label_idx, dataset_config.get("label_map"))

        # e.g. frames/label_0/signer0_label0_sample1/
        video_out_dir = out_dir / f"label_{label_idx}" / stem

        # Reuse earlier output unless overwrite=True
        meta_path = video_out_dir / "meta.json"
        if meta_path.exists() and not overwrite:
            with open(meta_path) as f:
                meta = json.load(f)
            stats["manifest"].append(meta)
            stats["processed"] += 1
            stats["label_distribution"][label_name] = (
                stats["label_distribution"].get(label_name, 0) + 1
            )
            continue

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

        video_out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = video_out_dir / "frames.npy"
        np.save(str(npy_path), frames)

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
            logger.info(f"  Progress: {i+1}/{len(video_files)} videos")

    manifest_path = Path(output_dir) / "frame_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(stats["manifest"], f, indent=2)

    if stats["frame_counts"]:
        stats["avg_frames"] = float(np.mean(stats["frame_counts"]))
        stats["min_frames_found"] = int(np.min(stats["frame_counts"]))
        stats["max_frames_found"] = int(np.max(stats["frame_counts"]))
    else:
        stats["avg_frames"] = 0

    logger.info(f"\n{'='*50}")
    logger.info(f"Frame extraction finished")
    logger.info(f"  Processed     : {stats['processed']}")
    logger.info(f"  Skipped short : {stats['skipped_short']}")
    logger.info(f"  Skipped error : {stats['skipped_error']}")
    if stats["frame_counts"]:
        logger.info(f"  Mean frames   : {stats['avg_frames']:.1f}")
        logger.info(f"  Frame range   : {stats['min_frames_found']} - {stats['max_frames_found']}")
    logger.info(f"  Manifest      : {manifest_path}")
    logger.info(f"{'='*50}\n")

    return stats


def _parse_filename(stem: str) -> Optional[Dict]:
    """
    Parse "signer0_label0_sample1" into {signer_idx: 0, label_idx: 0, sample_idx: 1}.
    Add other naming schemes here for other datasets.
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
