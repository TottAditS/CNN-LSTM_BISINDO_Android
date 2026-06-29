"""
FeatureExtraction/Scripts/keypoint_extractor.py

Step 1: Ekstraksi 21 keypoint per tangan menggunakan MediaPipe Hands.
- Input : frames .npy per video (output PreProcessing)
- Output: keypoint .npy per video, shape (T, 126)
           T   = jumlah frame
           126 = 2 tangan × 21 keypoint × 3 coords (x, y, z)

MediaPipe handle segalanya secara internal:
  1. Palm Detector → temukan area tangan di frame
  2. Hand Landmark → ekstrak 21 keypoint dari area itu
Tidak perlu preprocessing segmentasi tambahan.
"""

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

import cv2
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from FeatureExtraction.Config.feature_config import (
    PROCESSED_DIR, KEYPOINT_DIR,
    NUM_SIGNERS, NUM_LABELS, SAMPLES_PER_LABEL_PER_SIGNER,
    FEATURES_PER_FRAME, NUM_HANDS, KEYPOINTS_PER_HAND, COORDS_PER_KEYPOINT,
    MP_STATIC_IMAGE_MODE, MP_MAX_NUM_HANDS,
    MP_MIN_DETECTION_CONFIDENCE, MP_MIN_TRACKING_CONFIDENCE,
    NORMALIZE_KEYPOINTS, VERBOSE,
    get_keypoint_path, create_output_dirs,
)

mp_hands = mp.solutions.hands

WRIST_IDX = 0
MCP_IDX   = 9  # Middle finger MCP — untuk skala hand size


class KeypointExtractor:
    """
    Ekstrak 21 hand landmark per tangan per frame menggunakan MediaPipe.

    Output shape per video: (T, 126)
        - Tangan kiri  : index [0:63]
        - Tangan kanan : index [63:126]
        - Tangan tidak terdeteksi → zeros

    Normalisasi (jika NORMALIZE_KEYPOINTS=True):
        - Kurangi koordinat wrist → posisi relatif ke wrist
        - Bagi dengan jarak wrist→MCP9 → scale invariant
    """

    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=MP_STATIC_IMAGE_MODE,
            max_num_hands=MP_MAX_NUM_HANDS,
            min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
        )
        if VERBOSE:
            print(f"[KeypointExtractor] MediaPipe Hands siap.")
            print(f"  max_num_hands      : {MP_MAX_NUM_HANDS}")
            print(f"  detection_conf     : {MP_MIN_DETECTION_CONFIDENCE}")
            print(f"  tracking_conf      : {MP_MIN_TRACKING_CONFIDENCE}")
            print(f"  normalize_keypoints: {NORMALIZE_KEYPOINTS}")
            print(f"  features_per_frame : {FEATURES_PER_FRAME}")

    def _landmarks_to_array(self, hand_landmarks) -> np.ndarray:
        """MediaPipe NormalizedLandmarkList → numpy (63,)."""
        kpts = np.zeros(KEYPOINTS_PER_HAND * COORDS_PER_KEYPOINT, dtype=np.float32)
        for i, lm in enumerate(hand_landmarks.landmark):
            kpts[i * 3]     = lm.x
            kpts[i * 3 + 1] = lm.y
            kpts[i * 3 + 2] = lm.z
        return kpts

    def _normalize(self, kpts: np.ndarray) -> np.ndarray:
        """
        Normalisasi keypoint relatif ke wrist dan skala tangan.
        Hasil: koordinat dalam frame referensi tangan itu sendiri.
        """
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
        """
        Ekstrak keypoint dari satu frame BGR.
        Returns: (126,) float32 — zeros jika tidak terdeteksi.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results   = self.hands.process(frame_rgb)

        keypoints = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)

        if results.multi_hand_landmarks is None:
            return keypoints

        for hand_lm, hand_info in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            label = hand_info.classification[0].label  # "Left" atau "Right"
            kpts  = self._landmarks_to_array(hand_lm)
            kpts  = self._normalize(kpts)

            slot = 0 if label == "Left" else KEYPOINTS_PER_HAND * COORDS_PER_KEYPOINT
            keypoints[slot : slot + KEYPOINTS_PER_HAND * COORDS_PER_KEYPOINT] = kpts

        return keypoints

    def extract_sequence(self, frames: np.ndarray) -> np.ndarray:
        """
        Ekstrak keypoint dari sequence frame satu video.

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
        """Proses semua 1600 video dan simpan keypoint .npy."""
        create_output_dirs()

        total     = NUM_SIGNERS * NUM_LABELS * SAMPLES_PER_LABEL_PER_SIGNER
        processed = 0
        skipped   = 0
        errors    = 0
        low_detect = 0

        iterator = tqdm(total=total, desc="Ekstraksi keypoint", unit="video")

        for signer_id in range(NUM_SIGNERS):
            for label_id in range(NUM_LABELS):
                for sample_id in range(1, SAMPLES_PER_LABEL_PER_SIGNER + 1):

                    kpt_path = get_keypoint_path(signer_id, label_id, sample_id)
                    vid_name = f"signer{signer_id}_label{label_id}_sample{sample_id}"

                    if skip_existing and kpt_path.exists():
                        skipped += 1
                        iterator.update(1)
                        continue

                    frame_npy = PROCESSED_DIR / f"{vid_name}_frames.npy"
                    if not frame_npy.exists():
                        if VERBOSE:
                            print(f"  [SKIP] Frame tidak ada: {frame_npy}")
                        errors += 1
                        iterator.update(1)
                        continue

                    try:
                        frames    = np.load(frame_npy)      # (T, H, W, 3)
                        keypoints = self.extract_sequence(frames)

                        # Cek kualitas deteksi
                        detected = np.any(keypoints != 0, axis=1).sum()
                        rate     = detected / len(keypoints)
                        if rate < 0.5:
                            low_detect += 1
                            if VERBOSE:
                                print(f"  [WARN] {vid_name}: deteksi rendah "
                                      f"({rate*100:.0f}% frame)")

                        np.save(kpt_path, keypoints)
                        processed += 1

                    except Exception as e:
                        print(f"  [ERROR] {vid_name}: {e}")
                        errors += 1

                    iterator.update(1)

        iterator.close()
        self.close()

        print(f"\n[KeypointExtractor] Selesai:")
        print(f"  Diproses   : {processed}")
        print(f"  Dilewati   : {skipped}")
        print(f"  Error      : {errors}")
        print(f"  Low detect : {low_detect} video (< 50% frame terdeteksi)")
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
