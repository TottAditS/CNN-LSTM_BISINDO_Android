# 🏋️ Training Setup Guide

Panduan training & evaluasi untuk dua model di repo ini:
- **CNNsa-BiLSTM** (`Training/CNNsaBiLSTM/`) — proposed model, input keypoint MediaPipe.
- **MViTv2** (`Training/MViTv2/`) — baseline transformer pembanding (paper *Word-Level BISINDO:
  A Novel Video Indonesian Sign Language Dataset and Baseline Methods*), input raw video frame.

Kedua model dievaluasi di **video val/test yang identik** (split di-share lewat
`FeatureExtraction/Scripts/video_split.py`) supaya hasil `Training/Scripts/comparator.py`
apple-to-apple.

---

## 📁 Struktur Direktori

```
Training/
├── Config/
│   └── shared_config.py         ← RESULTS_DIR (SHARED oleh kedua model + comparator.py)
├── Scripts/
│   ├── comparator.py             ← Baca hasil evaluasi kedua model, cetak tabel perbandingan
│   └── device_utils.py           ← get_device() (SHARED): pilih GPU, warning jelas kalau jatuh ke CPU
│
├── CNNsaBiLSTM/                  ← Proposed model (keypoint MediaPipe)
│   ├── Config/train_config.py    ← Hyperparameter & path (import RESULTS_DIR dari shared_config)
│   ├── Models/cnnsa_bilstm.py    ← Arsitektur CNNsa-BiLSTM
│   ├── Scripts/
│   │   ├── trainer.py            ← Training loop (checkpoint, early stopping, LR schedule)
│   │   └── evaluator.py          ← Metrik, confusion matrix, benchmark CPU
│   └── Train.ipynb               ← Notebook controller (jalankan ini)
│
├── MViTv2/                       ← Baseline transformer (raw video, pretrained Kinetics-400)
│   ├── Config/mvit_config.py     ← Hyperparameter, freeze strategy, clip settings
│   ├── Models/mvit_v2.py         ← Wrapper torchvision mvit_v2_s + freeze/head replacement
│   ├── Scripts/
│   │   ├── mvit_dataset.py       ← VideoClipDataset (raw frame -> clip ternormalisasi)
│   │   ├── mvit_trainer.py       ← Training loop dengan AMP + differential LR
│   │   └── mvit_evaluator.py     ← Metrik, confusion matrix, benchmark CPU
│   └── TrainMViT.ipynb           ← Notebook controller (jalankan ini)
│
└── Output/                       ← Auto-generated saat run
    ├── cnnsa_bilstm/{checkpoints,logs}/
    ├── mvit_v2/{checkpoints,logs}/
    └── results/                  ← SHARED: dibaca comparator.py
        ├── evaluation_summary.json               (CNNsa-BiLSTM)
        ├── transformer_evaluation_summary.json   (MViTv2)
        ├── confusion_matrix_cnnsa_bilstm.png
        └── confusion_matrix_mvit_v2.png
```

---

## 🚀 Langkah 1: Persiapan Requirements

Gunakan Python 3.11 (3.10-3.12 juga aman). `torch` dan `torchvision` (dipakai kedua model, MViTv2
butuh `torchvision.models.video`) sudah ada di `Requirements.txt` root repo:
```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

> ⚠️ Wheel `cu128` (CUDA 12.8) **wajib** kalau GPU kamu Blackwell/RTX 50-series (compute capability
> sm_120) — wheel CUDA lama seperti `cu121` tidak punya kernel untuk arsitektur ini dan training akan
> gagal atau diam-diam jatuh ke CPU. Cek compute capability GPU: `nvidia-smi --query-gpu=compute_cap --format=csv`.
> Setelah install, verifikasi dengan `python -c "import torch; print(torch.cuda.is_available())"`, atau
> jalankan salah satu notebook — cell setup otomatis print nama GPU (lihat `Training/Scripts/device_utils.py`
> -> `get_device()`, dipakai kedua model) atau warning jelas kalau ternyata jatuh ke CPU.

Instalasi dependency visualisasi dan evaluasi (juga sudah di `Requirements.txt`):
```bash
pip install matplotlib seaborn scikit-learn tensorboard
```

---

## 🏃 Langkah 2: CNNsa-BiLSTM (Proposed Model)

Prasyarat: `FeatureExtraction/FeatureExtraction.ipynb` sudah dijalankan (butuh `*_sequences.npz`).

### Opsi A: Jupyter Notebook (disarankan)
```bash
cd Training/CNNsaBiLSTM
jupyter notebook Train.ipynb
```
Jalankan cell dari atas: cek dataset & device → model summary → training loop → evaluasi test split
→ plot loss/accuracy → confusion matrix.

### Opsi B: Command Line
```bash
python Training/CNNsaBiLSTM/Scripts/trainer.py     # training
python Training/CNNsaBiLSTM/Scripts/evaluator.py   # evaluasi -> evaluation_summary.json
```

Hyperparameter di **`Training/CNNsaBiLSTM/Config/train_config.py`**:
- **`batch_size`**: Default `64`. Kurangi jika OOM.
- **`learning_rate`**: Default `1e-3`.
- **`patience`** (Early Stopping): Default `10` epoch.
- **`hidden_size`**: Default `256`, dimensi internal BiLSTM.

---

## 🤖 Langkah 3: MViTv2 (Baseline Transformer)

Prasyarat: cuma butuh `PreProcessing/PreProcess.ipynb` (frame_manifest.json + frames.npy) —
**tidak perlu** FeatureExtraction, karena MViTv2 makan raw video frame langsung, bukan keypoint.

### Opsi A: Jupyter Notebook (disarankan)
```bash
cd Training/MViTv2
jupyter notebook TrainMViT.ipynb
```
Alur sama seperti `Train.ipynb`, ditambah cell terakhir yang memanggil `comparator.compare_models()`
untuk cetak tabel perbandingan vs CNNsa-BiLSTM (jalankan setelah CNNsa-BiLSTM juga dievaluasi).

### Opsi B: Command Line
```bash
python Training/MViTv2/Scripts/mvit_trainer.py     # fine-tune
python Training/MViTv2/Scripts/mvit_evaluator.py   # evaluasi -> transformer_evaluation_summary.json
```

Hyperparameter di **`Training/MViTv2/Config/mvit_config.py`**:
- **`FREEZE_UNTIL_BLOCK`**: Default `12` (dari 16 block). Blocks awal dibekukan (fitur generik
  Kinetics-400), blocks akhir + head di-fine-tune — dataset di sini kecil (~1000 klip) & VRAM
  terbatas (8GB), jadi freeze sebagian besar backbone lebih tahan overfit. Turunkan (mis. `8` atau
  `0`) untuk fine-tune lebih dalam / full fine-tune kalau VRAM & data cukup.
- **`TRAIN_CONFIG.batch_size`**: Default `4` (clip 16×224×224 berat di VRAM). Naikkan kalau GPU
  kamu lega.
- **`TRAIN_CONFIG.backbone_lr` / `head_lr`**: LR berbeda untuk backbone (kecil, sudah pretrained)
  vs head baru (besar, random init).
- **`TRAIN_CONFIG.use_amp`**: Default `True` — mixed precision, disarankan tetap aktif di GPU 8GB.

> ⚠️ MViTv2 jauh lebih berat dari CNNsa-BiLSTM (34.5M vs ~1-2M parameter, input video bukan
> keypoint) — training per-epoch jauh lebih lambat, dan model >131MB **bukan kandidat deploy ke
> Android** (beda dari CNNsa-BiLSTM yang memang didesain untuk itu). Benchmark CPU di evaluator
> murni untuk tabel perbandingan skripsi.

---

## 📊 Membandingkan Kedua Model

Setelah **keduanya** dievaluasi (baik lewat notebook atau command line):
```bash
python Training/Scripts/comparator.py
```
Baca `Training/Output/results/evaluation_summary.json` dan `transformer_evaluation_summary.json`,
cetak tabel accuracy / F1 / latensi CPU / ukuran model side-by-side.

---

## 📈 Monitoring dengan TensorBoard

```bash
tensorboard --logdir=Training/Output/cnnsa_bilstm/logs   # CNNsa-BiLSTM
tensorboard --logdir=Training/Output/mvit_v2/logs        # MViTv2
```
Buka `http://localhost:6006/` untuk melihat kurva loss, akurasi, dan learning rate.
