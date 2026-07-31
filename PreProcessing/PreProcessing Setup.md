# CNN-LSTM BISINDO — Preprocessing

## Struktur File

```
PreProcessing/
├── PreProcess.ipynb              ← Notebook utama (jalankan ini)
├── Config/
│   └── dataset_config.py         ← Konfigurasi dataset & preprocessing
└── Scripts/
    └── frame_extractor.py        ← Step 1: Video → Frames (Blur → Resize → Uniform Sampling)
```

> ℹ️ **Augmentasi & split train/val/test tidak dilakukan di sini.** Notebook ini hanya
> extract frame mentah. Keduanya dilakukan di `FeatureExtraction/FeatureExtraction.ipynb`
> pada level **keypoint** (bukan pixel) — lihat `FeatureExtraction Setup.md`. Alasannya:
> model final memakai keypoint MediaPipe (T, 126), dan augmentasi fotometrik pada pixel
> (brightness/contrast) tidak berpengaruh sama sekali pada keypoint yang sudah dinormalisasi.

## Cara Pakai

### 1. Setup path dataset

Edit `Config/dataset_config.py`:
```python
"WLBisindo": {
    "raw_video_dir": "path/ke/folder/WLBisindo",  # ← Ganti ini
    ...
}
```

### 2. Jalankan notebook

Buka `PreProcess.ipynb` dan jalankan semua cell dari atas ke bawah.

### 3. Menambah dataset baru

Cukup tambahkan entry baru di `DATASET_CONFIGS` dalam `dataset_config.py`:
```python
"NamaDatasetBaru": {
    "raw_video_dir": "...",
    "output_dir": "...",
    ...
}
```
Lalu di notebook, ganti:
```python
ACTIVE_DATASET = "NamaDatasetBaru"
```

## Format nama file yang didukung

Default: `signer{N}_label{N}_sample{N}.mp4`

Untuk format lain, tambahkan parser di `Scripts/frame_extractor.py` → fungsi `_parse_filename()`.

## Output

```
Datasets/WLBisindo_Processed/
├── frames/                    ← frame per video (.npy), bertingkat per label
└── frame_manifest.json        ← metadata semua video (signer, label, sample, path frame)
```

**Next step**: `FeatureExtraction/FeatureExtraction.ipynb` — ekstraksi keypoint, split, dan augmentasi.
