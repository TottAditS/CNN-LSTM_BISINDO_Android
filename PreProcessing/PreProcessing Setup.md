# CNN-LSTM BISINDO — Preprocessing

## Struktur File

```
preprocessing/
├── PreProcess.ipynb              ← Notebook utama (jalankan ini)
├── config/
│   └── dataset_config.py         ← Konfigurasi dataset & preprocessing
└── scripts/
    ├── frame_extractor.py         ← Step 1: Video → Frames
    ├── augmentor.py               ← Step 2: Augmentation + Padding
    └── dataloader_builder.py      ← Step 3: Split + Simpan .npz
```

## Cara Pakai

### 1. Setup path dataset

Edit `config/dataset_config.py`:
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

Untuk format lain, tambahkan parser di `scripts/frame_extractor.py` → fungsi `_parse_filename()`.

## Output

```
data/processed/WLBisindo/
├── frames/                    ← frame per video
├── augmented/                 ← sequence augmented
└── dataloader/
    ├── train.npz              ← shape: (N, T, H, W, C)
    ├── val.npz
    ├── test.npz
    ├── label_map.json
    └── split_info.json
```
