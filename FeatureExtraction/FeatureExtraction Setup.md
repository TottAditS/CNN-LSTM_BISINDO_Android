# FeatureExtraction — Setup & Guide

## Struktur

```
FeatureExtraction/
├── Config/
│   └── feature_config.py       ← Auto-detect ROOT, semua setting terpusat (termasuk AUGMENTATION_CONFIG)
├── Scripts/
│   ├── keypoint_extractor.py   ← Step 1: MediaPipe → 21 keypoint per tangan
│   ├── keypoint_augmentor.py   ← Dipakai Step 2: augmentasi flip/rotate/zoom pada keypoint
│   └── sequence_builder.py     ← Step 2: split train/val/test → augmentasi (train saja) → (N, 30, 126) → .npz
└── FeatureExtraction.ipynb     ← Notebook controller — jalankan dari sini
```

## Install

Sudah termasuk di `Requirements.txt` root repo — install dari sana, jangan `pip install mediapipe`
polos:
```bash
pip install mediapipe==0.10.21
```
> ⚠️ **Wajib versi ini, bukan asal terbaru.** Mulai mediapipe 0.10.30 (termasuk 1.0.0), Google
> menghapus total API `mediapipe.solutions` yang dipakai `keypoint_extractor.py` di bawah
> (`mp.solutions.hands.Hands(...)`), diganti Tasks API baru yang butuh file model `.task` terpisah.
> 0.10.21 adalah versi terakhir yang masih punya `mp.solutions` — sudah diverifikasi lewat install
> nyata. `numpy` juga ikut dipin `<2` di `Requirements.txt` karena mediapipe versi ini mensyaratkannya.

## Pipeline

```
frames.npy (dari PreProcessing)
  → [Step 1] MediaPipe Hands  → (T, 126) per video → keypoints/*.npy
  → [Step 2] Sequence Builder:
       a. Split video ke train/val/test (stratified per label, sebelum augmentasi)
       b. Augmentasi keypoint (flip/rotate/zoom) — hanya video yang masuk train
       c. Pad/crop temporal ke 30 frame → sequences/*.npz
```

MediaPipe handle deteksi tangan secara internal — tidak perlu segmentasi manual.

Augmentasi dilakukan di sini (bukan di PreProcessing) karena model final memakai keypoint
MediaPipe, bukan raw pixel — augmentasi fotometrik (brightness/contrast) tidak berpengaruh
ke koordinat keypoint yang sudah dinormalisasi, jadi hanya transformasi geometris (flip,
rotate, zoom) yang dipakai. Augmentasi hanya diterapkan setelah split ditentukan, supaya
val/test selalu berisi data asli dan tidak bocor dari train.

## Cara Menjalankan

Buka `FeatureExtraction.ipynb`, jalankan cell per cell dari atas.

### Step 1 (~30–60 menit)
- MediaPipe deteksi tangan di tiap frame → 21 keypoint × 2 tangan × 3 coords
- Output: `Datasets/WLBisindo_Features/keypoints/*.npy`, shape `(T, 126)` (belum diaugmentasi)

### Step 2 (~1–5 menit)
- Stratified split video per label → train/val/test
- Augmentasi (flip/rotate/zoom) untuk video train — lihat `AUGMENTATION_CONFIG` di `feature_config.py`
- Pad/crop temporal ke 30 frame
- Output: `sequences/train_sequences.npz` (original + augmented), `val_sequences.npz`, `test_sequences.npz` (original saja)

## Output Akhir

```
Datasets/WLBisindo_Features/sequences/
  train_sequences.npz   ← X: (N_train, 30, 126), y: (N_train,) — original + augmented
  val_sequences.npz     ← original saja
  test_sequences.npz    ← original saja
```

## Konfigurasi Penting

| Setting | Value | Keterangan |
|---|---|---|
| `NUM_SIGNERS` | 5 | signer0–signer4 sesuai `Datasets/WLBisindo` saat ini |
| `NUM_LABELS` | 32 | label0–label31 sesuai `Datasets/WLBisindo` saat ini — update jika dataset bertambah |
| `SEQUENCE_LENGTH` | 30 | Fixed frame per sequence |
| `FEATURES_PER_FRAME` | 126 | 2 × 21 × 3 |
| `NORMALIZE_KEYPOINTS` | True | Subtract wrist, scale by hand size |
| `CENTER_CROP_TEMPORAL` | True | Crop tengah jika T > 30 |
| `MP_MIN_DETECTION_CONFIDENCE` | 0.5 | Turunkan jika banyak miss |
| `AUGMENTATION_CONFIG.enabled` | True | Matikan jika tidak mau augmentasi sama sekali |
| `AUGMENTATION_CONFIG.techniques` | flip, rotation, zoom | Hanya diterapkan ke train |
| `AUGMENTATION_CONFIG.augment_multiplier` | 2 | 1 original + 2 augmented = 3x data train |
