# 🤟 CNNsa-BiLSTM for BISINDO Recognition
### Real-Time Indonesian Sign Language Recognition on Android

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-red)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green)
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
│   Classifier            │  ← FC(512→256) → ReLU → Dropout → FC(256→40) → Softmax
│                         │    Output: 40 kelas kata BISINDO
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
│    Uniform Sampling (15fps) → Augmentation → Padding     │
│    Output: frame_manifest.json + frames.npy per video    │
│    Notebook: PreProcessing/PreProcess.ipynb              │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ 2. FEATURE EXTRACTION                                    │
│    MediaPipe Hands → Palm Detection → 21 Keypoints       │
│    × 3 (x,y,z) × 2 tangan = 126 features/frame          │
│    Temporal Pad/Crop → 30 frame fixed length             │
│    Output: train/val/test_sequences.npz  (N, 30, 126)    │
│    Notebook: FeatureExtraction/FeatureExtraction.ipynb   │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ 3. MODEL TRAINING & EVALUATION                           │
│    CNNsa-BiLSTM (proposed model)                         │
│    Metrics: Accuracy, F1, Precision, Recall,             │
│             Inference Time, Model Size                   │
│    Notebook: Training/Train.ipynb                        │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ 4. EXPORT & DEPLOYMENT                                   │
│    PyTorch → ONNX → Android APK                          │
│    Real-time inference via CameraX + MediaPipe           │
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
│   │   ├── augmented/                  ← sequence augmented per label
│   │   ├── frame_manifest.json         ← metadata semua video
│   │   └── augmented_manifest.json     ← metadata setelah augmentasi
│   └── WLBisindo_Features/             ← Output feature extraction (auto-generated)
│       ├── keypoints/                  ← keypoint.npy per video (T, 126)
│       └── sequences/                  ← train/val/test_sequences.npz (N, 30, 126)
│
├── 📂 PreProcessing/                   ← Step 1: Video → Frame Sequences
│   ├── Config/
│   │   └── dataset_config.py           ← Konfigurasi dataset & preprocessing
│   ├── Scripts/
│   │   ├── frame_extractor.py          ← Ekstraksi frame dari video
│   │   ├── augmentor.py                ← Augmentasi & padding sequence
│   │   └── dataloader_builder.py       ← Split train/val/test → .npz
│   ├── PreProcess.ipynb                ← Notebook controller (jalankan ini)
│   └── PreProcessing Setup.md
│
├── 📂 FeatureExtraction/               ← Step 2: Frame → Keypoint Sequences
│   ├── Config/
│   │   └── feature_config.py           ← Konfigurasi feature extraction
│   ├── Scripts/
│   │   ├── keypoint_extractor.py       ← MediaPipe → 21 keypoints per tangan
│   │   └── sequence_builder.py         ← Build (N, 30, 126) sequences
│   ├── FeatureExtraction.ipynb         ← Notebook controller (jalankan ini)
│   └── FeatureExtraction Setup.md
│
├── 📂 Training/                        ← Step 3: Train & Evaluate Model
│   ├── Config/
│   │   └── train_config.py             ← Hyperparameter & path terpusat
│   ├── Models/
│   │   └── cnnsa_bilstm.py             ← Model CNNsa-BiLSTM (proposed)
│   ├── Scripts/
│   │   ├── trainer.py                  ← Training loop + checkpoint + early stopping
│   │   ├── evaluator.py                ← Metrik, confusion matrix, benchmark
│   │   └── comparator.py               ← Placeholder perbandingan model
│   ├── Train.ipynb                     ← Notebook controller (jalankan ini)
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
| Total Video | 1.600+ video |
| Format | `signer{N}_label{N}_sample{N}.mp4` |
| Jumlah Kelas | 40 kata BISINDO |
| Jumlah Signer | 4 signer |
| Sample per Kelas | Bervariasi (10–15 per signer) |
| Setelah Augmentasi | ~4.800 sequences (estimasi) |
| Split | 70% Train / 15% Val / 15% Test |

---

## ⚙️ Preprocessing Config

| Parameter | Value |
|---|---|
| Target FPS | 15 fps |
| Frame Size | 224 × 224 px |
| Sequence Length | 30 frames |
| Augmentasi | Flip, Brightness, Contrast, Rotation, Zoom |
| Augment Multiplier | 2 augmented + 1 original = 3x per video |
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
| Num Classes | 40 |
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
| Deep Learning | PyTorch 2.3.0 |
| GPU Training | NVIDIA RTX 4060 (CUDA 12.1) |
| Computer Vision | OpenCV, Pillow |
| Hand Keypoints | MediaPipe Hands |
| Model Export | ONNX |
| Android Inference | ONNX Runtime Mobile |
| Android Camera | CameraX |
| Experiment Tracking | TensorBoard |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/username/CNN-LSTM_BISINDO_Android.git
cd CNN-LSTM_BISINDO_Android
pip install -r Requirements.txt
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

### 4. Feature Extraction
```bash
cd FeatureExtraction
jupyter notebook FeatureExtraction.ipynb
```

### 5. Training
```bash
cd Training
jupyter notebook Train.ipynb
```

---

## 📋 Requirements

```
torch==2.3.0+cu121
torchvision==0.18.0+cu121
opencv-python>=4.8.0
mediapipe>=0.10.0
Pillow>=10.0.0
numpy>=1.24.0,<2.0.0
scikit-learn>=1.3.0
onnx>=1.15.0
onnxruntime>=1.17.0
```

Install PyTorch dengan GPU support:
```bash
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## 📚 References

- Tang et al. (2022) — CNN-LSTM-Attention untuk prediksi adaptif
- Aljabar & Suharjito (2020) — BISINDO recognition framework
- Kindy et al. (2025) — Mobile deployment untuk sign language recognition

---

## 👤 Author

**[Nama Kamu]**
Thesis Project — [Nama Universitas]
[Tahun]

---

> ⚠️ **Status**: Project sedang dalam tahap pengembangan aktif. Pipeline PreProcessing dan FeatureExtraction sudah selesai dibuat. Training sedang dalam proses setup.
