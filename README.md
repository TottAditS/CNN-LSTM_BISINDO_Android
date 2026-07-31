# 🤟 CNNsa-BiLSTM for BISINDO Recognition
### Real-Time Indonesian Sign Language Recognition on Android

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-red)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green)
![Android](https://img.shields.io/badge/Android-API%2026+-brightgreen)
![ONNX](https://img.shields.io/badge/Export-ONNX-purple)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

---

## 📖 Overview

Penelitian ini mengembangkan sistem pengenalan gesture dinamis **Bahasa Isyarat Indonesia (BISINDO)** berbasis deep learning yang diimplementasikan secara real-time pada perangkat Android. Model yang diusulkan adalah **CNN-Spatial Attention-BiLSTM-Attention (CNNsa-BiLSTM)** berbasis keypoint MediaPipe, yang dirancang untuk menyeimbangkan akurasi tinggi dan efisiensi komputasi agar dapat berjalan pada perangkat mobile dengan sumber daya terbatas.

> 📌 **Target**: Akurasi >90% dengan latency <100ms pada perangkat Android

---

## 🎯 Latar Belakang

Indonesia memiliki sekitar **1,92 juta individu Tuli**, namun hanya tersedia **~500 Juru Bahasa Isyarat (JBI) bersertifikat** hingga 2025. Ketimpangan ini menciptakan hambatan komunikasi signifikan di sektor pendidikan, kesehatan, dan layanan publik. Penelitian ini hadir sebagai solusi teknologi yang memungkinkan komunikasi mandiri dan real-time melalui smartphone.

---

## 🏗️ Arsitektur Model

Model CNNsa-BiLSTM menggunakan **keypoint MediaPipe** `(30, 126)` sebagai input — bukan raw image — sehingga jauh lebih ringan dan efisien untuk deployment Android.

```
Input Keypoints (30 frames × 126 features)
   126 = 2 tangan × 21 keypoint × 3 coords (x, y, z)
         │
         ▼
┌─────────────────────────┐
│   Per-frame CNN Block   │  ← Conv1D layers ekstraksi fitur dari 126 keypoint
│   (shared across frames)│    Output: (30, 256)
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Spatial Attention     │  ← Fokus pada fitur keypoint yang paling relevan
│   (feature-wise)        │    Output: (30, 256) — weighted features
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Bidirectional LSTM    │  ← Pemodelan temporal maju & mundur
│   (2 layers, hidden=256)│    Output: (30, 512)
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Temporal Attention    │  ← Fokus pada frame paling informatif dari 30 frame
│   (frame-wise)          │    Output: (512,) — context vector
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Classifier            │  ← FC(512→256) → ReLU → Dropout → FC(256→32) → Softmax
│                         │    Output: 32 kelas kata BISINDO
└─────────────────────────┘
```

---

## 🔄 Pipeline Lengkap

```
📹 Raw Video Dataset
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ 1. PRE-PROCESSING                                        │
│    Scan Video → Blur Filter → Resize (224×224)           │
│    Uniform Sampling (15fps)                              │
│    Output: frame_manifest.json + frames.npy per video    │
│    Notebook: PreProcessing/PreProcess.ipynb               │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ 2. FEATURE EXTRACTION                                    │
│    MediaPipe Hands → Palm Detection → 21 Keypoints       │
│    × 3 (x,y,z) × 2 tangan = 126 features/frame          │
│    Split Train/Val/Test → Augmentation (flip/rotate/zoom,│
│    train saja) → Temporal Pad/Crop → 30 frame fixed      │
│    Output: train/val/test_sequences.npz  (N, 30, 126)    │
│    Notebook: FeatureExtraction/FeatureExtraction.ipynb   │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ 3a. TRAINING — CNNsa-BiLSTM (proposed model)             │
│    Input: keypoint sequences dari Step 2                 │
│    Metrics: Accuracy, F1, Precision, Recall,             │
│             Inference Time, Model Size                   │
│    Notebook: Training/CNNsaBiLSTM/Train.ipynb            │
└──────────────────────────────────────────────────────────┘
         │
         ▼ (paralel, bisa jalan tanpa Step 2 — baca frames.npy langsung dari Step 1)
┌──────────────────────────────────────────────────────────┐
│ 3b. TRAINING — MViTv2 (baseline transformer)             │
│    Input: raw video clip (16 frame, 224×224) dari Step 1 │
│    Fine-tune dari bobot pretrained Kinetics-400           │
│    Notebook: Training/MViTv2/TrainMViT.ipynb              │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ 3c. PERBANDINGAN MODEL                                   │
│    Training/Scripts/comparator.py — video val/test SAMA  │
│    untuk kedua model (split di-share), tabel akurasi/F1/  │
│    latensi/ukuran model side-by-side                     │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ 4. EXPORT & DEPLOYMENT (CNNsa-BiLSTM saja)                │
│    PyTorch → ONNX → Android APK                          │
│    Real-time inference via CameraX + MediaPipe           │
│    (MViTv2 >131MB — baseline pembanding riset saja,       │
│     bukan kandidat deploy mobile)                         │
└──────────────────────────────────────────────────────────┘
```

---

## 📱 Android App Flow

```
┌──────────────────────────┐
│  ┌────────────────────┐  │
│  │                    │  │
│  │   [LIVE CAMERA]    │  │
│  │                    │  │
│  │    hand tracked    │  │
│  │  • landmark dots • │  │
│  └────────────────────┘  │
│                          │
│  Status: Ready...        │
│  ████████░░  24/30       │
│                          │
│  Result:  TERIMA KASIH   │
│                          │
│  History:                │
│  HALO | APA | KABAR      │
│                          │
│     [  RECORD  ]         │
└──────────────────────────┘
```

**Alur Inferensi Android:**
1. User tekan **RECORD**
2. CameraX merekam frame kamera
3. MediaPipe Hands ekstrak 126 keypoint per frame (secara langsung di Android)
4. Setelah 30 frame terkumpul → ONNX Runtime inferensi
5. Tampil **1 kata** hasil terjemahan

---

## 📁 Struktur Direktori

```
CNN-LSTM_BISINDO_Android/
│
├── 📂 Datasets/
│   ├── WLBisindo/                      ← Raw video (*.mp4)
│   ├── WLBisindo_Processed/            ← Output preprocessing (auto-generated)
│   │   ├── frames/                     ← frames.npy per video (bertingkat per label)
│   │   └── frame_manifest.json         ← metadata semua video
│   └── WLBisindo_Features/             ← Output feature extraction (auto-generated)
│       ├── keypoints/                  ← keypoint.npy per video asli (T, 126)
│       └── sequences/                  ← train/val/test_sequences.npz (N, 30, 126)
│                                          (train = original + augmented, val/test = original saja)
│
├── 📂 PreProcessing/                   ← Step 1: Video → Frame Sequences
│   ├── Config/
│   │   └── dataset_config.py           ← Konfigurasi dataset & preprocessing
│   ├── Scripts/
│   │   └── frame_extractor.py          ← Ekstraksi frame dari video
│   ├── PreProcess.ipynb                ← Notebook controller (jalankan ini)
│   └── PreProcessing Setup.md
│
├── 📂 FeatureExtraction/               ← Step 2: Frame → Keypoint Sequences (khusus CNNsa-BiLSTM)
│   ├── Config/
│   │   └── feature_config.py           ← Konfigurasi feature extraction & augmentasi
│   ├── Scripts/
│   │   ├── keypoint_extractor.py       ← MediaPipe → 21 keypoints per tangan
│   │   ├── keypoint_augmentor.py       ← Augmentasi keypoint (flip/rotate/zoom), train saja
│   │   ├── video_split.py              ← Split train/val/test SHARED (dipakai juga MViTv2)
│   │   └── sequence_builder.py         ← Split → augmentasi → (N, 30, 126) sequences
│   ├── FeatureExtraction.ipynb         ← Notebook controller (jalankan ini)
│   └── FeatureExtraction Setup.md
│
├── 📂 Training/                        ← Step 3: Train, Evaluasi & Bandingkan Model
│   ├── Config/
│   │   └── shared_config.py            ← RESULTS_DIR (SHARED, dibaca comparator.py)
│   ├── Scripts/
│   │   ├── comparator.py               ← Tabel perbandingan CNNsa-BiLSTM vs MViTv2
│   │   └── device_utils.py             ← get_device() (SHARED oleh kedua model)
│   ├── CNNsaBiLSTM/                    ← Proposed model (input: keypoint)
│   │   ├── Config/train_config.py      ← Hyperparameter & path
│   │   ├── Models/cnnsa_bilstm.py      ← Arsitektur CNNsa-BiLSTM
│   │   ├── Scripts/{trainer.py, evaluator.py}
│   │   └── Train.ipynb                 ← Notebook controller
│   ├── MViTv2/                         ← Baseline transformer (input: raw video, pretrained Kinetics-400)
│   │   ├── Config/mvit_config.py       ← Hyperparameter, freeze strategy, clip settings
│   │   ├── Models/mvit_v2.py           ← Wrapper torchvision mvit_v2_s
│   │   ├── Scripts/{mvit_dataset.py, mvit_trainer.py, mvit_evaluator.py}
│   │   └── TrainMViT.ipynb             ← Notebook controller
│   └── Training Setup.md
│
├── 📂 Export/                          ← Step 4: PyTorch → ONNX (belum diimplementasi)
│
└── Requirements.txt
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Nama | WLBisindo (Word-Level BISINDO) |
| Total Video | 1.600 video |
| Format | `signer{N}_label{N}_sample{N}.mp4` |
| Jumlah Kelas | 32 kata BISINDO (`label0`–`label31`) |
| Jumlah Signer | 5 signer (`signer0`–`signer4`) — `signer0` baru merekam 12/32 kelas |
| Sample per Kelas | Bervariasi (10 atau 15 per signer) |
| Split | 70% Train / 15% Val / 15% Test (per label, sebelum augmentasi) |
| Augmentasi | Hanya di train split, level keypoint (flip/rotate/zoom), 3x data train |

> ⚠️ Target awal riset adalah 40 kata / 4 signer (lihat tabel Training Config). Dataset yang
> sudah terkumpul saat ini baru 32 kelas dari 5 signer. `NUM_LABELS`/`NUM_CLASSES` di semua
> config sudah disesuaikan ke 32 — update kembali ke 40 jika dataset sudah lengkap.

---

## ⚙️ Preprocessing Config

| Parameter | Value |
|---|---|
| Target FPS | 15 fps |
| Frame Size | 224 × 224 px |
| Output Format | `.npy` per video + `frame_manifest.json` |

## 🔀 Feature Extraction — Split & Augmentation Config

| Parameter | Value |
|---|---|
| Sequence Length | 30 frames |
| Split | 70% Train / 15% Val / 15% Test (stratified per label, **sebelum** augmentasi) |
| Augmentasi | Flip, Rotation, Zoom — level keypoint, **hanya train split** |
| Augment Multiplier | 2 augmented + 1 original = 3x data train |
| Output Format | `.npz` (NumPy compressed) |

---

## 🏋️ Training Config

| Parameter | Value |
|---|---|
| Input Shape | (N, 30, 126) |
| Feature Dim (CNN out) | 256 |
| BiLSTM Hidden Size | 256 (× 2 = 512 bidirectional) |
| BiLSTM Layers | 2 |
| Dropout | 0.3 |
| Num Classes | 32 (sesuai dataset saat ini — lihat bagian Dataset) |
| Batch Size | 64 |
| Learning Rate | 1e-3 |
| Optimizer | Adam (weight_decay=1e-4) |
| LR Scheduler | ReduceLROnPlateau |
| Max Epochs | 100 |
| Early Stopping | patience=10 |

---

## 📈 Model Evaluation (Belum Dijalankan)

| Metrik | CNNsa-BiLSTM (Proposed) |
|---|---|
| Accuracy | - |
| F1-Score (weighted) | - |
| Inference Time CPU | - |
| Model Size (MB) | - |
| Android Compatible | ✅ |

> *Hasil akan diisi setelah eksperimen selesai*

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|---|---|
| Deep Learning | PyTorch 2.11.0 |
| GPU Training | NVIDIA GPU + CUDA 12.8 wheel (wajib untuk GPU Blackwell / RTX 50-series, backward-compatible untuk GPU lama) |
| Computer Vision | opencv-contrib-python, Pillow |
| Hand Keypoints | MediaPipe Hands (CPU-only di Windows — GPU delegate belum didukung Google di platform ini) |
| Model Export | ONNX |
| Android Inference | ONNX Runtime Mobile |
| Android Camera | CameraX |
| Experiment Tracking | TensorBoard |

> ⚠️ **GPU note**: training PyTorch (kedua model) otomatis pakai GPU kalau `torch.cuda.is_available()` = True
> (lihat `Training/Scripts/device_utils.py` -> `get_device()`, dipakai `CNNsaBiLSTM/Scripts/trainer.py` &
> `MViTv2/Scripts/mvit_trainer.py`, dipanggil di awal training/notebook dan akan print nama GPU + compute
> capability, atau warning jelas kalau jatuh ke CPU). Ekstraksi keypoint MediaPipe (`FeatureExtraction/`)
> selalu jalan di CPU terlepas dari ini — itu batasan resmi MediaPipe di Windows, bukan bug di repo ini.

---

## 🚀 Quick Start

### 1. Clone & Install
Gunakan Python 3.11 (3.10-3.12 juga aman — lihat `Requirements.txt` untuk detail kompatibilitas).
```bash
git clone https://github.com/username/CNN-LSTM_BISINDO_Android.git
cd CNN-LSTM_BISINDO_Android
pip install -r Requirements.txt
```

Verifikasi PyTorch benar-benar melihat GPU sebelum lanjut (penting kalau GPU kamu Blackwell/RTX 50-series —
lihat catatan CUDA 12.8 di `Requirements.txt`):
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

### 2. Siapkan Dataset
Letakkan video dataset di:
```
Datasets/WLBisindo/signer0_label0_sample1.mp4
                   signer0_label0_sample2.mp4
                   ...
```

### 3. Jalankan Preprocessing
```bash
cd PreProcessing
jupyter notebook PreProcess.ipynb
```

### 4. Feature Extraction (khusus CNNsa-BiLSTM — keypoint, split, augmentasi)
```bash
cd FeatureExtraction
jupyter notebook FeatureExtraction.ipynb
```

### 5. Training CNNsa-BiLSTM (proposed model)
```bash
cd Training/CNNsaBiLSTM
jupyter notebook Train.ipynb
```

### 6. Training MViTv2 (baseline transformer — cukup butuh Step 3, tidak perlu Step 4)
```bash
cd Training/MViTv2
jupyter notebook TrainMViT.ipynb
```
Cell terakhir `TrainMViT.ipynb` memanggil `comparator.compare_models()` untuk cetak tabel
perbandingan vs CNNsa-BiLSTM (jalankan setelah Step 5 juga selesai dievaluasi).

---

## 📋 Requirements

Daftar lengkap ada di `Requirements.txt` (`pip install -r Requirements.txt` install semuanya sekali jalan).
Inti dependency:

```
torch==2.11.0+cu128
torchvision==0.26.0+cu128
opencv-contrib-python==4.11.0.86  # JANGAN install opencv-python terpisah — konflik DLL di Windows
mediapipe==0.10.21                # dipin: versi >=0.10.30 (termasuk 1.0.0) menghapus API mp.solutions
Pillow>=10.0.0
numpy==1.26.4                     # dipin <2: mediapipe==0.10.21 mensyaratkan numpy<2
scikit-learn>=1.4.0
onnx>=1.17.0
onnxruntime>=1.20.0
```

> ⚠️ **mediapipe & numpy sengaja dipin ke versi lama** (bukan salah ketik): mulai mediapipe 0.10.30,
> Google menghapus total API `mediapipe.solutions` yang dipakai `keypoint_extractor.py` — hanya
> versi ≤0.10.21 yang masih punya `mp.solutions.hands`. Detail lengkap ada di komentar
> `Requirements.txt`.

Install PyTorch dengan GPU support (CUDA 12.8 — wajib untuk GPU Blackwell/RTX 50-series, tetap
backward-compatible untuk GPU NVIDIA generasi lama):
```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

---

## 📚 References

- Tang et al. (2022) — CNN-LSTM-Attention untuk prediksi adaptif
- Aljabar & Suharjito (2020) — BISINDO recognition framework
- Kindy et al. (2025) — Mobile deployment untuk sign language recognition
- **Word-Level BISINDO: A Novel Video Indonesian Sign Language Dataset and Baseline Methods**
  (*Procedia Computer Science*, vol. 269, pp. 249–258, 2025) — sumber dataset WLBisindo yang
  dipakai repo ini, dan referensi baseline transformer (SPOTER, Siformer, **MViTv2** — MViTv2
  yang diimplementasikan di `Training/MViTv2/` sebagai pembanding CNNsa-BiLSTM).

---

## 👤 Author

**[Nama Kamu]**
Thesis Project — [Nama Universitas]
[Tahun]

---

> ⚠️ **Status**: Project sedang dalam tahap pengembangan aktif. Pipeline PreProcessing, FeatureExtraction,
> dan Training (CNNsa-BiLSTM + baseline MViTv2, termasuk script perbandingan) sudah selesai dibuat secara
> kode — **belum pernah dijalankan end-to-end** pada dataset asli. Export ke ONNX/Android belum diimplementasi.
