"""
Hand keypoint extraction with MediaPipe Hands.

Input : frames.npy per video (PreProcessing output)
Output: keypoints .npy per video, shape (T, 126)
        126 = 2 hands x 21 landmarks x 3 coordinates (x, y, z)
"""

import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import cv2
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from FeatureExtraction.Config.feature_config import (
    PROCESSED_DIR, KEYPOINT_DIR,
    FEATURES_PER_FRAME, NUM_HANDS, KEYPOINTS_PER_HAND, COORDS_PER_KEYPOINT,
    MP_STATIC_IMAGE_MODE, MP_MAX_NUM_HANDS,
    MP_MIN_DETECTION_CONFIDENCE, MP_MIN_TRACKING_CONFIDENCE,
    NORMALIZE_KEYPOINTS, VERBOSE,
    get_keypoint_path, create_output_dirs,
)

mp_hands = mp.solutions.hands

WRIST_IDX = 0
MCP_IDX   = 9  # middle finger MCP, used for the hand-size scale


class KeypointExtractor:
    """
    Extracts 21 landmarks per hand per frame.

    Layout per frame (126 values):
        left hand  : [0:63]
        right hand : [63:126]
        a hand that is not detected stays zero

    With NORMALIZE_KEYPOINTS, each hand is shifted so the wrist is the origin and
    divided by the wrist-to-MCP9 distance.
    """

    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=MP_STATIC_IMAGE_MODE,
            max_num_hands=MP_MAX_NUM_HANDS,
            min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
        )
        if VERBOSE:
            print(f"[KeypointExtractor] MediaPipe Hands ready.")
            print(f"  max_num_hands      : {MP_MAX_NUM_HANDS}")
            print(f"  detection_conf     : {MP_MIN_DETECTION_CONFIDENCE}")
            print(f"  tracking_conf      : {MP_MIN_TRACKING_CONFIDENCE}")
            print(f"  normalize_keypoints: {NORMALIZE_KEYPOINTS}")
            print(f"  features_per_frame : {FEATURES_PER_FRAME}")

    def _landmarks_to_array(self, hand_landmarks) -> np.ndarray:
        """MediaPipe NormalizedLandmarkList -> numpy (63,)."""
        kpts = np.zeros(KEYPOINTS_PER_HAND * COORDS_PER_KEYPOINT, dtype=np.float32)
        for i, lm in enumerate(hand_landmarks.landmark):
            kpts[i * 3]     = lm.x
            kpts[i * 3 + 1] = lm.y
            kpts[i * 3 + 2] = lm.z
        return kpts

    def _normalize(self, kpts: np.ndarray) -> np.ndarray:
        """Express one hand relative to its wrist, scaled by hand size."""
        if not NORMALIZE_KEYPOINTS:
            return kpts

        kpts = kpts.copy().reshape(KEYPOINTS_PER_HAND, COORDS_PER_KEYPOINT)
        wrist = kpts[WRIST_IDX].copy()
        kpts -= wrist

        scale = np.linalg.norm(kpts[MCP_IDX])
        if scale > 1e-6:
            kpts /= scale

        return kpts.flatten()

    def extract_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Keypoints of one BGR frame as (126,) float32, zeros if no hand is found."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results   = self.hands.process(frame_rgb)

        keypoints = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)

        if results.multi_hand_landmarks is None:
            return keypoints

        for hand_lm, hand_info in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            label = hand_info.classification[0].label  # "Left" or "Right"
            kpts  = self._landmarks_to_array(hand_lm)
            kpts  = self._normalize(kpts)

            slot = 0 if label == "Left" else KEYPOINTS_PER_HAND * COORDS_PER_KEYPOINT
            keypoints[slot : slot + KEYPOINTS_PER_HAND * COORDS_PER_KEYPOINT] = kpts

        return keypoints

    def extract_sequence(self, frames: np.ndarray) -> np.ndarray:
        """
        Keypoints of every frame of one video.

        Args:
            frames: (T, H, W, 3) uint8 RGB

        Returns:
            keypoints: (T, 126) float32
        """
        T = frames.shape[0]
        keypoints = np.zeros((T, FEATURES_PER_FRAME), dtype=np.float32)

        for i in range(T):
            frame_bgr    = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
            keypoints[i] = self.extract_frame(frame_bgr)

        return keypoints

    def process_all(self, skip_existing: bool = True):
        """Extract keypoints for every video listed in frame_manifest.json."""
        create_output_dirs()

        manifest_path = PROCESSED_DIR / "frame_manifest.json"
        if not manifest_path.exists():
            print(f"[KeypointExtractor] ERROR: manifest not found at {manifest_path}")
            print("Run PreProcessing/PreProcess.ipynb first.")
            return

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        total     = len(manifest)
        processed = 0
        skipped   = 0
        errors    = 0
        low_detect = 0

        iterator = tqdm(manifest, desc="Keypoint extraction", unit="video")

        for item in iterator:
            signer_id = item["signer_idx"]
            label_id  = item["label_idx"]
            sample_id = item["sample_idx"]

            kpt_path = get_keypoint_path(signer_id, label_id, sample_id)
            vid_name = Path(item["video_name"]).stem  # "signerX_labelY_sampleZ"

            if skip_existing and kpt_path.exists():
                skipped += 1
                continue

            frame_npy = Path(item["frames_npy"])
            if not frame_npy.exists():
                if VERBOSE:
                    print(f"  [SKIP] Missing frames: {frame_npy}")
                errors += 1
                continue

            try:
                frames    = np.load(frame_npy)      # (T, H, W, 3)
                keypoints = self.extract_sequence(frames)

                detected = np.any(keypoints != 0, axis=1).sum()
                rate     = detected / len(keypoints)
                if rate < 0.5:
                    low_detect += 1
                    if VERBOSE:
                        print(f"  [WARN] {vid_name}: low detection rate "
                              f"({rate*100:.0f}% of frames)")

                np.save(kpt_path, keypoints)
                processed += 1

            except Exception as e:
                print(f"  [ERROR] {vid_name}: {e}")
                errors += 1

        # The caller owns this extractor and calls close() itself; closing twice
        # raises a ValueError inside MediaPipe.

        print(f"\n[KeypointExtractor] Done:")
        print(f"  Processed  : {processed}")
        print(f"  Skipped    : {skipped}")
        print(f"  Errors     : {errors}")
        print(f"  Low detect : {low_detect} videos (< 50% of frames detected)")
        print(f"  Output     : {KEYPOINT_DIR}")

    def close(self):
        self.hands.close()

    def __del__(self):
        try:
            self.hands.close()
        except Exception:
            pass


if __name__ == "__main__":
    extractor = KeypointExtractor()
    extractor.process_all(skip_existing=True)
