# FeatureExtraction — Setup & Guide

## Struktur

```
FeatureExtraction/
├── Config/
│   └── feature_config.py      ← Auto-detect ROOT, semua setting terpusat
├── Scripts/
│   ├── keypoint_extractor.py  ← Step 1: MediaPipe → 21 keypoint per tangan
│   └── sequence_builder.py    ← Step 2: (N, 30, 126) sequences + train/val/test split
└── FeatureExtraction.ipynb    ← Notebook controller — jalankan dari sini
```

## Install

```bash
pip install mediapipe
```

## Pipeline

```
frames.npy (dari PreProcessing)
  → [Step 1] MediaPipe Hands  → (T, 126) per video → keypoints/*.npy
  → [Step 2] Sequence Builder → (N, 30, 126) split  → sequences/*.npz
```

MediaPipe handle deteksi tangan secara internal — tidak perlu segmentasi manual.

## Cara Menjalankan

Buka `FeatureExtraction.ipynb`, jalankan cell per cell dari atas.

### Step 1 (~30–60 menit)
- MediaPipe deteksi tangan di tiap frame → 21 keypoint × 2 tangan × 3 coords
- Output: `Datasets/WLBisindo_Features/keypoints/*.npy`, shape `(T, 126)`

### Step 2 (~1–5 menit)
- Pad/crop temporal ke 30 frame
- Stratified split per label
- Output: `sequences/train_sequences.npz`, `val_sequences.npz`, `test_sequences.npz`

## Output Akhir

```
Datasets/WLBisindo_Features/sequences/
  train_sequences.npz   ← X: (N_train, 30, 126), y: (N_train,)
  val_sequences.npz
  test_sequences.npz
```

## Konfigurasi Penting

| Setting | Value | Keterangan |
|---|---|---|
| `SEQUENCE_LENGTH` | 30 | Fixed frame per sequence |
| `FEATURES_PER_FRAME` | 126 | 2 × 21 × 3 |
| `NORMALIZE_KEYPOINTS` | True | Subtract wrist, scale by hand size |
| `CENTER_CROP_TEMPORAL` | True | Crop tengah jika T > 30 |
| `MP_MIN_DETECTION_CONFIDENCE` | 0.5 | Turunkan jika banyak miss |
