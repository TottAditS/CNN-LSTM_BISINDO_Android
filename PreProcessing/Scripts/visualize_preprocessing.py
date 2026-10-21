"""
Documentation figures for the preprocessing step (not part of the pipeline).

Applies the same steps and parameters as frame_extractor.py to one sample video
and writes:
    PreProcessing/Output/charts/preprocessing_before_after.png
    PreProcessing/Output/charts/preprocessing_frame_sampling.png
"""
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PreProcessing.Config.dataset_config import DATASET_CONFIGS, PREPROCESSING_CONFIG

OUT_DIR = Path(__file__).resolve().parents[2] / "PreProcessing" / "Output" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_VIDEO = Path(DATASET_CONFIGS["WLBisindo"]["raw_video_dir"]) / "signer0_label0_sample1.mp4"

frame_cfg = PREPROCESSING_CONFIG["frame"]


def load_raw_frame(video_path: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = min(frame_index, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    return frame, total  # BGR, uint8


def build_before_after_figure():
    total_frames_info = int(cv2.VideoCapture(str(SAMPLE_VIDEO)).get(cv2.CAP_PROP_FRAME_COUNT))
    mid_idx = total_frames_info // 2
    raw_bgr, total = load_raw_frame(SAMPLE_VIDEO, mid_idx)

    # Same steps as frame_extractor.extract_frames_from_video
    blurred_bgr = cv2.GaussianBlur(raw_bgr, tuple(frame_cfg["blur_kernel"]), frame_cfg["blur_sigma"])
    resized_bgr = cv2.resize(blurred_bgr, (frame_cfg["resize"][1], frame_cfg["resize"][0]))
    final_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    # Conversion for display only (matplotlib expects RGB), not a pipeline step
    raw_display = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    blurred_display = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB)

    panels = [
        (raw_display, f"1. Raw frame\n{raw_bgr.shape[1]}x{raw_bgr.shape[0]} px"),
        (blurred_display, f"2. Gaussian blur\nkernel {frame_cfg['blur_kernel']}"),
        (final_rgb, f"3. Resized, RGB (final input)\n{frame_cfg['resize'][1]}x{frame_cfg['resize'][0]} px"),
    ]
    # Width ratios follow each panel's aspect ratio so the square panel is not stretched
    width_ratios = [img.shape[1] / img.shape[0] for img, _ in panels]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), gridspec_kw={"width_ratios": width_ratios})
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"Frame preprocessing, sample {SAMPLE_VIDEO.name} (frame {mid_idx} of {total})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = OUT_DIR / "preprocessing_before_after.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


def build_frame_sampling_figure():
    cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, round(original_fps / frame_cfg["target_fps"]))

    sampled_frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            sampled_frames.append(frame)
            if len(sampled_frames) >= frame_cfg["max_frames"]:
                break
        frame_idx += 1
    cap.release()

    show_n = min(8, len(sampled_frames))
    step = max(1, len(sampled_frames) // show_n)
    picks = sampled_frames[::step][:show_n]

    fig, axes = plt.subplots(1, len(picks), figsize=(2.1 * len(picks), 2.6))
    for i, (ax, frame) in enumerate(zip(axes, picks)):
        rgb = cv2.cvtColor(cv2.resize(frame, (128, 128)), cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(f"#{i * step}", fontsize=9)
        ax.axis("off")

    fig.suptitle(
        f"Uniform temporal sampling, {original_fps:.1f} fps to {frame_cfg['target_fps']} fps "
        f"({len(sampled_frames)} of {total_frames} frames kept)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = OUT_DIR / "preprocessing_frame_sampling.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    assert SAMPLE_VIDEO.exists(), f"sample video not found: {SAMPLE_VIDEO}"
    build_before_after_figure()
    build_frame_sampling_figure()
