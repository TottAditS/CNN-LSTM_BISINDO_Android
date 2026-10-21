"""
Documentation figures for the feature extraction step (not part of the pipeline).

Shows MediaPipe Hands detections and the effect of keypoint normalization,
using the same class and parameters as keypoint_extractor.py. Writes:
    FeatureExtraction/Output/charts/feature_extraction_landmarks.png
    FeatureExtraction/Output/charts/feature_extraction_normalization.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PreProcessing.Config.dataset_config import DATASET_CONFIGS, PREPROCESSING_CONFIG
from FeatureExtraction.Config.feature_config import (
    MP_STATIC_IMAGE_MODE, MP_MAX_NUM_HANDS,
    MP_MIN_DETECTION_CONFIDENCE, MP_MIN_TRACKING_CONFIDENCE,
    KEYPOINTS_PER_HAND, COORDS_PER_KEYPOINT,
)
from FeatureExtraction.Scripts.keypoint_extractor import KeypointExtractor, WRIST_IDX, MCP_IDX

OUT_DIR = Path(__file__).resolve().parents[2] / "FeatureExtraction" / "Output" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_VIDEO = Path(DATASET_CONFIGS["WLBisindo"]["raw_video_dir"]) / "signer0_label0_sample1.mp4"
frame_cfg = PREPROCESSING_CONFIG["frame"]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def get_preprocessed_frame(frame_index: int) -> np.ndarray:
    """Same steps as PreProcessing/Scripts/frame_extractor.py: blur -> resize."""
    cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = min(frame_index, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_index}")
    frame = cv2.GaussianBlur(frame, tuple(frame_cfg["blur_kernel"]), frame_cfg["blur_sigma"])
    frame = cv2.resize(frame, (frame_cfg["resize"][1], frame_cfg["resize"][0]))
    return frame  # BGR, 224x224, as expected by KeypointExtractor.extract_frame


def build_landmark_figure(frame_bgr: np.ndarray):
    hands = mp_hands.Hands(
        static_image_mode=MP_STATIC_IMAGE_MODE,
        max_num_hands=MP_MAX_NUM_HANDS,
        min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
    )
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    annotated = frame_rgb.copy()
    n_hands = 0
    if results.multi_hand_landmarks:
        n_hands = len(results.multi_hand_landmarks)
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(
                annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
    hands.close()

    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    axes[0].imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input: preprocessed frame\n(224x224)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(annotated)
    axes[1].set_title(f"MediaPipe Hands output\n{n_hands} hand(s) detected, 21 landmarks per hand", fontsize=10)
    axes[1].axis("off")

    fig.suptitle(f"Keypoint extraction with MediaPipe Hands, sample {SAMPLE_VIDEO.name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = OUT_DIR / "feature_extraction_landmarks.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")
    return results


def build_normalization_figure(frame_bgr: np.ndarray):
    """Plot the raw and normalized keypoints of one hand, computed with KeypointExtractor."""
    extractor = KeypointExtractor()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = extractor.hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        print("[WARN] no hand detected in this frame, skipping the normalization figure")
        extractor.close()
        return

    hand_landmarks = results.multi_hand_landmarks[0]
    raw_kpts = extractor._landmarks_to_array(hand_landmarks).reshape(KEYPOINTS_PER_HAND, COORDS_PER_KEYPOINT)
    norm_kpts = extractor._normalize(raw_kpts.flatten()).reshape(KEYPOINTS_PER_HAND, COORDS_PER_KEYPOINT)
    extractor.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].scatter(raw_kpts[:, 0], -raw_kpts[:, 1], c=range(KEYPOINTS_PER_HAND), cmap="viridis", s=40)
    axes[0].scatter(raw_kpts[WRIST_IDX, 0], -raw_kpts[WRIST_IDX, 1], c="red", s=80, marker="x", label="wrist (idx 0)")
    axes[0].set_title("Before normalization\n(image coordinates, 0-1)", fontsize=10)
    axes[0].set_aspect("equal")
    axes[0].legend(fontsize=8)

    axes[1].scatter(norm_kpts[:, 0], -norm_kpts[:, 1], c=range(KEYPOINTS_PER_HAND), cmap="viridis", s=40)
    axes[1].scatter(0, 0, c="red", s=80, marker="x", label="wrist (origin)")
    axes[1].set_title("After normalization\n(relative to wrist, scaled by wrist-MCP9 distance)", fontsize=10)
    axes[1].set_aspect("equal")
    axes[1].legend(fontsize=8)

    fig.suptitle("Effect of keypoint normalization (21 landmarks, one hand)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = OUT_DIR / "feature_extraction_normalization.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    assert SAMPLE_VIDEO.exists(), f"sample video not found: {SAMPLE_VIDEO}"
    frame = get_preprocessed_frame(frame_index=40)
    build_landmark_figure(frame)
    build_normalization_figure(frame)
